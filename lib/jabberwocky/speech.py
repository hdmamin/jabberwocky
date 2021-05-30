import os
import shlex

from htools.meta import Callback, handle_interrupt


class SpeakingStatusCallback(Callback):

    def setup(self, func):
        pass

    def on_begin(self, func, inputs, output=None):
        inputs['self'].is_speaking = True

    def on_end(self, func, inputs, output=None):
        inputs['self'].is_speaking = False


class Speaker:
    """Pyttsx3 package has issues with threads on Mac. Messed around with
    subprocesses a bit too but ultimately decided to just use built in MAC OS
    functionality.
    """

    def __init__(self, voice='karen', rate=135, newline_pause=300):
        self.voice = voice
        self.rate = rate
        self.newline_pause = newline_pause
        self.cmd_prefix = f'say -v {self.voice} -r {self.rate} '
        self.is_speaking = False

    @handle_interrupt(cbs=SpeakingStatusCallback())
    def speak(self, text):
        os.system(self.cmd_prefix + self._format_text(text))

    def _format_text(self, text):
        return shlex.quote(text)\
                    .replace('\n', f'[[slnc {self.newline_pause}]]')
