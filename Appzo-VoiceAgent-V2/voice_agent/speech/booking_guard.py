"""Block unverified booking claims before any part of the sentence reaches TTS."""
import re


class BookingClaimGuard:
    def __init__(self):
        self.pending = ""

    def push(self, text, final=False):
        self.pending += text
        out = []
        while True:
            match = re.search(r"[.!?](?:\s|$)", self.pending)
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
        if re.search(r"\b(scheduled|booked|confirmed|reserved|will (?:call|reach out|contact))\b", sentence, re.I):
            return "Your requested callback time still needs confirmation from the team. "
        return sentence
