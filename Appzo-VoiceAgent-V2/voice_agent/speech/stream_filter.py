"""Strip wire markers across token boundaries before TTS."""
import re

class SpeechStreamFilter:
    def __init__(self):
        self.pending = ""

    def push(self, delta, final=False):
        self.pending += delta
        self.pending = re.sub(r"(?:OK|END|NO)\s*\|", "", self.pending)
        hold = re.search(r"(?:E(?:N(?:D)?)?|O(?:K)?|N(?:O)?)\s*$", self.pending)
        end = hold.start() if hold and not final else len(self.pending)
        result, self.pending = self.pending[:end], self.pending[end:]
        return result
