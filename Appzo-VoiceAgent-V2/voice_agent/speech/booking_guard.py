"""Block unverified booking claims before any part of the sentence reaches TTS."""
import re


class BookingClaimGuard:
    def __init__(self):
        self.pending = ""

    def push(self, text, final=False):
        self.pending += text
        out = []
        while True:
            # The runtime explicitly prompts the model to begin with a short,
            # independently speakable clause. Inspect and release those
            # clauses instead of buffering an entire long sentence. Booking
            # claims such as "I will arrange," are still caught in that first
            # clause before any text reaches TTS.
            match = re.search(r"[,;:.!?](?:\s|$)", self.pending)
            if not match:
                break
            sentence, self.pending = self.pending[:match.end()], self.pending[match.end():]
            out.append(self.check(sentence))
        if final and self.pending:
            out.append(self.check(self.pending))
            self.pending = ""
        return "".join(out)

    @staticmethod
    def check(sentence):
        # This guard is intentionally narrow enough not to rewrite ordinary
        # acknowledgements (for example, "noted, you need five people"), but
        # broad enough to catch the unsupported commitments observed in live
        # calls: "I have noted the meeting", "I can arrange a connect", and
        # "I will connect with you tomorrow". No scheduling tool exists in
        # this runtime, so none of these claims may reach TTS.
        # Deterministic callback copy already states the required limitation;
        # do not rewrite it merely because it says that a preference was
        # recorded. This keeps the explicit "not a confirmed booking" wording.
        if re.search(r"\bnot a confirmed booking\b|\bconfirm availability\b", sentence, re.I):
            return sentence
        claim = re.search(
            r"\b(?:scheduled|booked|confirmed|reserved)\b|"
            r"\b(?:i\s+)?(?:will|can|shall|am going to|['’]ll)\s+"
            r"(?:call|reach out|contact|connect|arrange|schedule|book)\b|"
            r"\b(?:i\s+)?(?:have|['’]ve)\s+(?:arranged|scheduled|booked|confirmed)\b|"
            r"\b(?:noted|recorded)\s+(?:the\s+)?(?:meeting|appointment|call|callback)\b|"
            r"\barrange(?:d|ment)?\s+(?:a|the\s+)?(?:meeting|appointment|call|callback|connect|follow-?up)\b",
            sentence,
            re.I,
        )
        if claim:
            return "Your requested follow-up time needs confirmation from the team. "
        return sentence
