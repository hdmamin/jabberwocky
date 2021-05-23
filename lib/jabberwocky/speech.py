import os
import shlex


class Speaker:
    """Pyttsx3 package has issues with threads on Mac. Messed around with
    subprocesses a bit too but ultimately decided to just use built in MAC OS
    functionality.
    """

    def __init__(self, voice='karen', rate=120):
        self.voice = voice
        self.rate = rate

    def speak(self, text):
        os.system(self._format_text(text))

    def _format_text(self, text):
        return f'say -v {self.voice} -r {self.rate} {shlex.quote(text)}'
