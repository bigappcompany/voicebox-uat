from dataclasses import dataclass, field
import re


@dataclass(frozen=True)
class KnowledgeRecord:
    tenant_id: str; agent_id: str; knowledge_version: str; document_id: str; text: str; risk_class: str = "LOW_PUBLIC"
    questions: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class KnowledgeMatch:
    record: KnowledgeRecord
    confidence: float
    reason: str


class TenantKnowledgeIndex:
    """In-memory, tenant-keyed retrieval. Replace the scorer, not this boundary."""
    def __init__(self) -> None: self._records: dict[tuple[str, str, str], list[KnowledgeRecord]] = {}

    def add(self, record: KnowledgeRecord) -> None:
        key = (record.tenant_id, record.agent_id, record.knowledge_version)
        records = self._records.setdefault(key, [])
        # Bootstrap may reuse a compiled bundle for many calls. Keep the
        # tenant index immutable by document identity instead of accumulating
        # duplicate passages per call.
        records[:] = [item for item in records if item.document_id != record.document_id]
        records.append(record)

    _stop = frozenset({"a", "an", "the", "do", "does", "is", "are", "you", "your", "we", "our", "what", "which", "kind", "of"})

    @classmethod
    def _tokens(cls, text: str) -> set[str]:
        return {word for word in re.findall(r"\w+", text.casefold()) if word not in cls._stop and len(word) > 1}

    def search_matches(self, *, tenant_id: str, agent_id: str, knowledge_version: str, query: str, top_k: int = 3) -> list[KnowledgeMatch]:
        normalized = " ".join(re.findall(r"\w+", query.casefold()))
        words = self._tokens(query)
        records = self._records.get((tenant_id, agent_id, knowledge_version), [])
        ranked: list[KnowledgeMatch] = []
        for record in records:
            aliases = [" ".join(re.findall(r"\w+", item.casefold())) for item in record.questions]
            if normalized and normalized in aliases:
                ranked.append(KnowledgeMatch(record, 1.0, "exact_alias"))
                continue
            alias_scores = []
            for alias in record.questions:
                alias_words = self._tokens(alias)
                if alias_words:
                    alias_scores.append(len(words & alias_words) / len(words | alias_words))
            text_words = self._tokens(record.text)
            coverage = len(words & text_words) / max(1, len(words))
            alias_score = max(alias_scores, default=0.0)
            score = min(1.0, max(alias_score * .95, coverage * .72))
            if score >= .2:
                ranked.append(KnowledgeMatch(record, score, "alias_overlap" if alias_score >= coverage else "content_overlap"))
        return sorted(ranked, key=lambda item: item.confidence, reverse=True)[:top_k]

    def search(self, *, tenant_id: str, agent_id: str, knowledge_version: str, query: str, top_k: int = 3) -> list[KnowledgeRecord]:
        return [
            match.record for match in self.search_matches(
                tenant_id=tenant_id, agent_id=agent_id,
                knowledge_version=knowledge_version, query=query, top_k=top_k,
            )
        ]
