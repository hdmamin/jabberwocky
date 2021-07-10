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
        self.is_speaking = False

    @property
    def cmd_prefix(self):
        """Must recompute in case user changes voice."""
        return f'say -v {self.voice} -r {self.rate} '

    @cmd_prefix.deleter
    def cmd_prefix(self):
        raise RuntimeError('cmd_prefix cannot be deleted.')

    @cmd_prefix.setter
    def cmd_prefix(self, voice):
        raise RuntimeError('cmd_prefix is read only.')

    @handle_interrupt(cbs=SpeakingStatusCallback())
    def speak(self, text):
        os.system(self.cmd_prefix + self._format_text(text))

    def _format_text(self, text):
        """For some reason `say` still interprets character following dash as
        a flag even after shlex.quote, so we add a backslash as well. I'm not
        sure if it actually escapes it or if the say program just ignores it,
        but so far it seems to work.

        Some speakers (e.g. "daniel") read "no." (either case) as "number".
        Speaker treats commas similarly to periods. If changing this logic,
        be mindful of spaces: we wouldn't want to change "This is a piano." to
        "This is a pia no." (ignore commas vs. periods: point is we don't
        want to insert a pause in the middle of the word "piano").
        """
        return shlex.quote(text)\
                    .replace('-', '\-')\
                    .replace('\n', f'[[slnc {self.newline_pause}]]')\
                    .replace('No.', 'No,')\
                    .replace(' no.', ' no,')
