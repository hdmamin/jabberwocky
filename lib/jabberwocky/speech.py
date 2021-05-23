import os
import shlex


class Speaker:
    """Pyttsx3 package has issues with threads on Mac. Messed around with
    subprocesses a bit too but ultimately decided to just use built in MAC OS
    functionality.
    """

    def __init__(self, voice='karen', rate=125, newline_pause=300):
        self.voice = voice
        self.rate = rate
        self.newline_pause = newline_pause
        self.cmd_prefix = f'say -v {self.voice} -r {self.rate} '

    def speak(self, text):
        os.system(self.cmd_prefix + self._format_text(text))

    def _format_text(self, text):
        return shlex.quote(text)\
                    .replace('\n', f'[[slnc {self.newline_pause}]]')
