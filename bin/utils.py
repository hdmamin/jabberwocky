"""General utilities used in our GUI that aren't callbacks. Note: this relies
on the SPEAKER variable defined in main.py (main.py makes this variable
available after importing utils - just making note of it here since pycharm
will show errors since SPEAKER is undefined at definition time).
"""

from contextlib import contextmanager as ctx_manager
from dearpygui.core import *
from dearpygui.simple import *
from nltk.tokenize import sent_tokenize
from threading import Thread
import time


@ctx_manager
def label_above(name, visible_name=None):
    """Create a dearpygui item with a label above it instead of to the right.

    Parameters
    ----------
    name: str
        The dearpygui name of the item in question.
    visible_name: str or None
        If provided, this should be the string you want to be displayed in the
        GUI as a label. If None, your item will have no label (i.e. this is a
        way to suppress the appearance of the default label on the right of the
        element.)
    """
    if visible_name: add_text(visible_name)
    try:
        yield
    finally:
        set_item_label(name, '')


def read_response(response, data):
    """Read response if desired. Threads allow us to interrupt speaker if user
    checks a checkbox. This was surprisingly difficult - I settled on a
    partial solution that can only quit after finishing saying a
    sentence/line, so there may be a bit of a delayed response after asking
    to interrupt.

    Parameters
    ----------
    response: str
        Text to read.
    data: dict
        Forwarded from callback. Must contain key 'interrupt_id' which tells
        us the dearpygui name of the checkbox to monitor in case the user asks
        to interrupt speech.
    """
    show_item(data['interrupt_id'])
    errors = []
    thread = Thread(target=monitor_interrupt_checkbox,
                    args=(data['interrupt_id'], errors))
    thread.start()
    try:
        for sent in sent_tokenize(response):
            for chunk in sent.split('\n\n'):
                SPEAKER.speak(chunk)
                if errors:
                    set_value(data['interrupt_id'], False)
                    raise errors[0]
    except RuntimeError:
        # Checkbox monitor raises this type of error if the user asks to
        # interrupt. Would need to use multiple break statements and an else
        # clause in the inner for loop otherwise (without it, a chunk of text
        # containing a double newline but no period would only break out of the
        # inner loop when an exception is raised).
        pass
    hide_item(data['interrupt_id'])
    thread.join()


def monitor_speaker(speaker, name, wait=1, quit_after=None, debug=False):
    """Track when speaker is speaking (run this function in a separate thread).
    Originally this was an attempt to implement speech interruption, but
    eventually I settled on a method where the interrupt button is only present
    during speech anyway so this check isn't necessary.

    Parameters
    ----------
    speaker: jabberwocky.speech.Speaker
    name: str
        Makes it easier to track which monitor is speaking.
    wait: int
        How frequently to check if the speaker is speaking.
    quit_after: int or None
        Max run time for the monitor. I feel like this shouldn't be necessary
        but IIRC threads weren't always closing otherwise (reasons unknown?) so
        I added this in for easier debugging.
    debug: bool
        If True, print status updates to console.
    """
    start = time.perf_counter()
    while True:
        if debug: print(f'[{name}] speaking: ' + str(speaker.is_speaking))
        time.sleep(wait)
        if quit_after and time.perf_counter() - start > quit_after:
            if debug: print(f'[{name}]: quitting due to time exceeded')
            break


def monitor_interrupt_checkbox(box_id, errors, wait=1, quit_after=None,
                               error_type=RuntimeError):
    """Track when the interrupt option is checked (run this function in a
    separate thread). Couldn't figure out a way to check this with a button
    (is_item_clicked seems to check only at that exact instant) so we use a
    slightly clunkier-looking checkbox.

    Parameters
    ----------
    box_id: str
        Name of dearpygui checkbox to monitor.
    errors: list
        List to track errors in main thread. It starts out empty but this
        function will append True (arbitrarily) if the checkbox of interest is
        checked. The main thread can then periodically check if the list
        remains empty. This is a workaround solution to the trickier task of
        propagating an exception from a thread to the main thread (which I
        read may not be a good idea anyway).
    wait: int
        How frequently to check if the speaker is speaking. A value of 2 means
        we'd check once every 2 seconds.
    quit_after: int or None
        Max run time for the monitor. I feel like this shouldn't be necessary
        but IIRC threads weren't always closing otherwise (reasons unknown?) so
        I added this in for easier debugging.
    error_type: type
        Exception class. This determines the type of error that will be
        appended to the list of errors if the user wants to interrupt the
        speech.
    """
    start = time.perf_counter()
    while True:
        if get_value(box_id):
            errors.append(error_type('User interrupted speaking session.'))
            print('Checkbox monitor quitting due to checkbox selection.')
            break
        time.sleep(wait)
        if not SPEAKER.is_speaking:
            print('Checkbox monitor quitting due to end of speech.')
            break
        if quit_after and time.perf_counter() - start > quit_after:
            print(f'Checkbox monitor quitting due to time exceeded.')
            break


def stream_text(text, chunk_size=3):
    """Generator that yields chunks of a string. GPT3 provides streaming mode
    but gpt-neo doesn't and we sometimes also want to display error messages or
    mocked results as if they are in streaming mode. This helps us do that.
    """
    yield from (text[i:i+chunk_size] for i in range(0, len(text), chunk_size))


def stream(text_or_gen):
    """Wrapper around stream_text(). Input can either be a string OR a
    generator (as returned by query_gpt3() when stream=True).
    """
    if isinstance(text_or_gen, str):
        yield from stream_text(text_or_gen)
    else:
        yield from text_or_gen
