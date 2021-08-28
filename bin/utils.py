"""General utilities used in our GUI that aren't callbacks. Note: this relies
on the SPEAKER variable defined in main.py (main.py makes this variable
available after importing utils - just making note of it here since pycharm
will show errors since SPEAKER is undefined at definition time).
"""

from contextlib import contextmanager as ctx_manager
import ctypes
from dearpygui.core import *
from dearpygui.simple import *
from functools import wraps
from nltk.tokenize import sent_tokenize
import threading
from threading import Thread
import _thread
import time

from htools.meta import coroutine


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


def read_response(response, data, errors=None, hide_on_exit=True):
    """Deprecated in favor of read_response_coro. Note that this version
    runs the checkbox monitor while the coroutine version leaves that to the
    concurrent_typing_speaking function (wasn't working in dearpygui otherwise,
    though the core non-dearpygui functionality worked in ipython).

    Read response if desired. Threads allow us to interrupt speaker if user
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
    # Careful: can't use "errors or []" here because we want to allow the user
    # to pass in an empty list and watch it to see if errors are appended.
    if errors is None:
        errors = []
    thread = Thread(target=monitor_interrupt_checkbox,
                    args=(data['interrupt_id'], errors, SPEAKER))
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
    if hide_on_exit: hide_item(data['interrupt_id'])
    thread.join()


@coroutine
def read_response_coro(data, errors=None, hide_on_exit=True):
    # Must send None as an extra last item so that this coroutine knows when
    # we're done sending in new tokens so it can check for any unread text.
    def _exit(data, hide_on_exit=True):
        set_value(data['interrupt_id'], False)
        if hide_on_exit:
            hide_item(data['interrupt_id'])
        SPEAKER.end_session()

    show_item(data['interrupt_id'])
    if errors is None:
        # Careful: can't use "errors or []" here because we want to allow the
        # user to pass in an empty list and watch it to see if errors are
        # appended.
        errors = []

    # Watch out for user requests to interrupt speaker. This will be joined in
    # _exit().
    text = ''
    # Must start manually otherwise contextmanager never exits. Seems to be
    # dearpygui-related since it works in ipython.
    SPEAKER.start_session()
    while not errors:
        token = yield
        if token is None:
            if sents: SPEAKER.speak(sents[0])
            _exit(data, hide_on_exit)
        else:
            text += token
            sents = sent_tokenize(text)
            if len(sents) > 1:
                for chunk in sents[0].split('\n\n'):
                    SPEAKER.speak(chunk)
                text = text.replace(sents[0], '', 1)
        if errors:
            _exit(data, hide_on_exit)


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


def monitor_interrupt_checkbox(box_id, errors, obj, attr='is_speaking',
                               wait=1, initial_grace_period=0, quit_after=None,
                               error_type=RuntimeError):
    """Track when the interrupt option is checked (run this function in a
    separate thread). Couldn't figure out a way to check this with a button
    (is_item_clicked seems to check only at that exact instant) so we use a
    slightly clunkier-looking checkbox.

    Parameters
    ----------
    box_id: str
        Name of dearpygui checkbox to monitor.
    errors: list or None
        List to track errors in main thread. It starts out empty but this
        function will append True (arbitrarily) if the checkbox of interest is
        checked. The main thread can then periodically check if the list
        remains empty. This is a workaround solution to the trickier task of
        propagating an exception from a thread to the main thread (which I
        read may not be a good idea anyway).

        If None, an error will be raised directly instead of simply being
        appended to a list. You must use PropagatingThread to be able to catch
        this.
    obj: any
        The object doing whatever it is that may need to be interrupted.
        Often speech_recognition.Recognizer or jabberwocky.speech.Speaker.
        It must have an attribute that tells us when it starts and finishes
        (see `attr` param).
    attr: str
        Attribute of obj to monitor. When this is Falsy (indicating the event
        to potentially interrupt is no longer ongoing), the monitor will quit.
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
            print('Checkbox monitor quitting due to checkbox selection.')
            if errors is None:
                raise error_type('User interrupted session.')
            else:
                errors.append(error_type('User interrupted session.'))
            break
        time.sleep(wait)
        if not getattr(obj, attr):
            if time.perf_counter() - start > initial_grace_period:
                print('Checkbox monitor quitting because obj finished '
                      'speaking/listening.')
                break
            else:
                print('Checkbox monitor not exiting because we\'re still in '
                      'the initial grace period.')
        if quit_after and time.perf_counter() - start > quit_after:
            print(f'Checkbox monitor quitting due to time exceeded.')
            break


def stream_chars(text, chunk_size=3):
    """Generator that yields chunks of a string. GPT3 provides streaming mode
    but gpt-neo doesn't and we sometimes also want to display error messages or
    mocked results as if they are in streaming mode. This helps us do that.
    """
    yield from (text[i:i+chunk_size] for i in range(0, len(text), chunk_size))


def stream_words(text):
    """Like stream_chars but splits on spaces. Realized stream_chars was a bad
    idea because we risk giving SPEAKER turns like
    "This is over. W" and "hat are you doing next?", neither of which would be
    pronounced as intended. We yield with a space for consistency with the
    other streaming interfaces which require no further postprocessing.
    """
    for word in text.split(' '):
        yield word + ' '


def stream(text_or_gen):
    """Wrapper around stream_text(). Input can either be a string OR a
    generator (as returned by query_gpt3() when stream=True).
    """
    if isinstance(text_or_gen, str):
        yield from stream_words(text_or_gen)
    else:
        yield from text_or_gen


class CoroutinableThread(Thread):
    """Lets us send values to a coroutine running in a thread. In this context
    we're not necessarily doing anything async with the coroutine - it's just
    a nice way to process new values.
    """

    def __init__(self, target, queue, args=(), kwargs=None):
        # Target should be a coroutine like read_response_coro.
        super().__init__(target=target, args=args, kwargs=kwargs)
        self.target = target(*args, **(kwargs or {}))
        self.queue = queue

    def run(self):
        while True:
            val = self.queue.get()
            self.target.send(val)
            # Must do this after send so our coroutine gets the sentinel.
            if val is None: return


def interrupt_on_complete(meth):
    """Decorator that powers PropagatingThread. We do this here rather than
    interrupting directly from run() method because run must return in order
    for thread to stop, but of course we can't do anything directly from the
    method after returning.
    """
    @wraps(meth)
    def wrapper(*args, **kwargs):
        res = meth(*args, **kwargs)
        if args[0].exception and args[0].raise_immediately:
            print('INTERRUPTING')
            _thread.interrupt_main()
        return res
    return wrapper


class PropagatingThread(Thread):
    """Thread that will raise an exception in the calling thread as soon as one
    occurs in the worker thread. You must use a KeyboardInterrupt or
    BaseException in your try/except block since that is the only type of
    exception we can raise. If you don't need the exception to be raised
    immediately, you can set raise_immediately=False and it won't be propagated
    until the thread is joined.

    thread.join() also returns the value returned by the thread's target
    function.

    Partially based on version here, though that can only raise on join():
    https://stackoverflow.com/questions/2829329/catch-a-threads-exception-in-the-caller-thread
    """

    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs=None, *, daemon=None,
                 raise_immediately=False):
        """
        Parameters
        ----------
        raise_immediately: bool
            If True, raise any exception as soon as it occurs. In order to
            propagate this to the calling thread, we are forced to use a
            KeyboardInterrupt. No information about the actual exception will
            be propagated. If False, the actual exception will be raised on
            the call to thread.join().
        """
        super().__init__(group=group, target=target, name=name,
                         args=args, kwargs=kwargs, daemon=daemon)
        self.raise_immediately = raise_immediately
        self.exception = None
        self.result = None

    @interrupt_on_complete
    def run(self):
        try:
            self.result = self._target(*self._args, **self._kwargs)
        except Exception as e:
            self.exception = e

    def join(self, timeout=None):
        super().join(timeout)
        if not self.raise_immediately and self.exception:
            raise self.exception
        return self.result


def interrupt(thread, exc_type=RuntimeError):
    """Interrupt a running thread. Apparently doesn't work when waiting for
    time.sleep().

    From https://gist.github.com/liuw/2407154 with minor tweaks.

    Parameters
    ----------
    thread: threading.Thread
        A thread which is currently alive.
    exc_type: type
        Exception class to raise when interrupting the thread.
    """
    if thread.ident not in threading._active:
        raise ValueError('Thread not found.')

    n_stopped = ctypes.pythonapi.PyThreadState_SetAsyncExc(
        ctypes.c_long(thread.ident),
        ctypes.py_object(exc_type)
    )

    if n_stopped == 0:
        raise ValueError('Invalid thread ID.')
    if n_stopped > 1:
        ctypes.pythonapi.PyThreadState_SetAsyncExc(thread.ident, 0)
        raise SystemError('PyThreadState_SetAsyncExc failed')
