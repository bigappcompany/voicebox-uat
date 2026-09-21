try:
    import pytest
except ModuleNotFoundError:  # The repository's default suite is unittest.
    class _Mark:
        @staticmethod
        def asyncio(function):
            return function

    class _PytestFallback:
        mark = _Mark()

        @staticmethod
        def fixture(function):
            return function

    pytest = _PytestFallback()
from types import SimpleNamespace
from unittest.mock import Mock, patch
from voice_agent.runtime.metrics import TurnMetrics
from main import BotStartedSpeakingFrame, FrameDirection, LiveLatencyObserver, TTSStoppedFrame
from voice_agent.runtime.latency_breakdown import LatencyBreakdown, LatencyContribution
from voice_agent.runtime.intents import CanonicalIntentModel
from voice_agent.speech.booking_guard import BookingClaimGuard
from voice_agent.llm.context_adapter import ConversationContextAdapter, HostedContext


@pytest.fixture
def mock_state():
    state = Mock()
    state.metrics = TurnMetrics(turn_id=1, route="hosted")
    return state


def make_observer(controller=None):
    return LiveLatencyObserver(
        controller or Mock(), tts_transport="websocket", session=Mock(), call_origin_at=0.0
    )


def test_observer_initialization():
    observer = make_observer()
    assert isinstance(observer._reported_turn_ids, set)
    assert isinstance(observer._samples, dict)


def test_turn_metrics_has_context_fields():
    metrics = TurnMetrics(turn_id=1)
    assert hasattr(metrics, "pipecat_context_read_ms")
    assert hasattr(metrics, "context_selection_ms")
    assert hasattr(metrics, "context_token_estimation_ms")
    assert hasattr(metrics, "prompt_build_ms")


def test_flux_derived_metrics():
    metrics = TurnMetrics(turn_id=1)
    metrics.last_voiced_at = 100.0
    metrics.eager_eot_at = 101.5
    metrics.turn_committed_at = 102.0
    
    observer = make_observer()
    record = observer._record(metrics, LatencyBreakdown(
        turn_id=1, measured_from="last_voiced", started_at=100.0, ended_at=105.0, contributions=()
    ))
    assert record["raw_speech_end_to_eager_eot_ms"] == 1500.0
    assert record["eager_eot_to_hard_eot_ms"] == 500.0
    assert record["raw_speech_end_to_hard_eot_ms"] == 2000.0


def test_speculation_derived_metrics():
    metrics = TurnMetrics(turn_id=1)
    metrics.speculative_started_at = 100.0
    metrics.first_safe_text_at = 101.0
    metrics.spec_tts_started_at = 101.1
    metrics.spec_tts_pcm_ready_at = 101.5
    metrics.turn_committed_at = 102.0
    
    observer = make_observer()
    record = observer._record(metrics, LatencyBreakdown(
        turn_id=1, measured_from="last_voiced", started_at=100.0, ended_at=105.0, contributions=()
    ))
    assert record["spec_start_to_first_safe_ms"] == 1000.0
    assert record["spec_start_to_pcm_ready_ms"] == 400.0
    assert record["spec_tts_savings_ms"] == 500.0


def test_boolean_interpreter_yes():
    model = CanonicalIntentModel()
    assert model._parse_boolean("that is true yeah") == "yes"
    assert model._parse_boolean("yes please") == "yes"
    assert model._parse_boolean("sure") == "yes"


def test_boolean_interpreter_no():
    model = CanonicalIntentModel()
    assert model._parse_boolean("yeah no dont do that") == "no"
    assert model._parse_boolean("nope") == "no"
    assert model._parse_boolean("not interested") == "no"


def test_boolean_interpreter_ambiguous():
    model = CanonicalIntentModel()
    assert model._parse_boolean("I am not sure") == "ambiguous"


def test_booking_guard_catches_unsupported():
    text = "I will arrange the follow-up for you tomorrow."
    result = BookingClaimGuard.check(text)
    assert "confirmation from the team" in result


def test_booking_guard_allows_safe():
    text = "Got it, you want a meeting. What day works for you?"
    result = BookingClaimGuard.check(text)
    assert "confirmation from the team" not in result


def test_latency_breakdown_formatting():
    metrics = TurnMetrics(turn_id=1)
    metrics.turn_committed_at = 100.0
    metrics.bot_started_at = 102.0
    
    breakdown = LatencyBreakdown(
        turn_id=1, measured_from="hard_eot", started_at=100.0, ended_at=102.0, contributions=(
            LatencyContribution("test", "test_label", "test_kind", "test_owner", 100.0, 102.0),
        )
    )
    lines = breakdown.turn_contribution_lines()
    assert any("TOTAL" in line for line in lines)
    assert any("test_label" in line for line in lines)


@patch("voice_agent.llm.context_adapter.ConversationContextAdapter._dialogue")
def test_hosted_context_timings(mock_dialogue):
    mock_dialogue.return_value = [{"role": "user", "content": "hi"}]
    adapter = ConversationContextAdapter(Mock(), Mock())
    context = adapter.build_hosted_context(current_user_text="hi")
    assert context.pipecat_read_ms >= 0.0
    assert context.selection_ms >= 0.0
    assert context.token_estimation_ms >= 0.0


def test_report_deduplication(mock_state):
    observer = make_observer()
    observer._reported_turn_ids.add(1)
    # Should not create task
    with patch("asyncio.create_task") as mock_task:
        observer._schedule_report_if_not_reported(mock_state)
        mock_task.assert_not_called()


@pytest.mark.asyncio
async def test_bot_started_is_only_a_fallback_when_pcm_metrics_are_enabled(mock_state):
    controller = Mock()
    controller._state = mock_state
    latency_observer = make_observer(controller)
    mock_state.metrics.tts_requested_at = 1.0
    event = SimpleNamespace(
        direction=FrameDirection.DOWNSTREAM, frame=BotStartedSpeakingFrame()
    )
    with patch("asyncio.create_task") as create_task:
        await latency_observer.on_push_frame(event)
        create_task.assert_not_called()


@pytest.mark.asyncio
async def test_final_stop_reports_when_output_pcm_was_not_observed(mock_state):
    controller = Mock()
    controller._state = mock_state
    latency_observer = make_observer(controller)
    mock_state.metrics.tts_requested_at = 1.0
    event = SimpleNamespace(
        direction=FrameDirection.DOWNSTREAM, frame=TTSStoppedFrame()
    )
    with patch("asyncio.create_task") as create_task:
        await latency_observer.on_push_frame(event)
        create_task.assert_called_once()
