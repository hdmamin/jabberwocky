"""Module to help us interact with mac's speech command. This lets the GUI read
responses out loud.
"""

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

    def __init__(self, voice='karen', rate=5, newline_pause=300):
        """

        Parameters
        ----------
        voice
        rate: int
            Determines speaker speed. Scale of 0-10 where 10 is fastest and 0
            is slowest.
        newline_pause
        """
        self.voice = voice
        self.newline_pause = newline_pause
        self.is_speaking = False

        # _rate will be updated automatically by setter method.
        self._min_rate = 120
        self.rate = rate
        self._rate = None

    @property
    def rate(self):
        return self._rate

    @rate.setter
    def rate(self, val):
        assert 0 <= val <= 10, 'You should choose a rate in [0, 10].'
        self._rate = self._min_rate + 100*(val / 10)

    @property
    def cmd_prefix(self):
        """Must recompute in case user changes voice."""
        return f'say -v {self.voice} -r {self._rate} '

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
