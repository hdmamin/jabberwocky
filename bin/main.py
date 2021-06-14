from contextlib import contextmanager as ctx_manager
from dearpygui.core import *
from dearpygui.simple import *
from nltk.tokenize import sent_tokenize
import os
import speech_recognition as sr
import time
from threading import Thread

from htools.core import tolist, select, eprint
from htools.meta import params
from htools.structures import IndexedDict
from jabberwocky.openai_utils import PromptManager, query_gpt3, query_gpt_neo
from jabberwocky.speech import Speaker
from jabberwocky.utils import most_recent_filepath, img_dims, _img_dims


os.chdir('../')
MANAGER = PromptManager(verbose=False)
NAME2TASK = IndexedDict({
    'Punctuate': 'punctuate',
    'Conversation': 'conversation',
    'Translate': 'translate',
    'Default': 'default',
    'Debate': 'debate',
    'Summarize': 'tldr',
    'Analyze Writing': 'analyze_writing',
    'Explain Like I\'m 5': 'eli',
    'Explain Machine Learning': 'simplify_ml',
    'Machine Learning Abstract Writer': 'ml_abstract',
    'How To': 'how_to',
    'MMA': 'mma',
    'Dates (debug)': 'short_dates',
    'Math (debug)': 'shortest'
})
MODEL_NAMES = ['gpt3', 'gpt-neo 2.7B', 'gpt-neo 1.3B', 'gpt-neo 125M', 'naive']
SPEAKER = Speaker(newline_pause=400)


@ctx_manager
def label_above(name, visible_name=None):
    if visible_name: add_text(visible_name)
    try:
        yield
    finally:
        set_item_label(name, '')


def transcribe_callback(sender, data):
    """Triggered when user hits the Record button.

    data keys:
        - target_id
        - show_during_ids
        - show_after_ids
    """
    set_value(data['target_id'], '')
    show_during = data.get('show_during_ids', [])
    for id_ in show_during:
        show_item(id_)

    # Record until pause.
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        audio = recognizer.listen(source)
    try:
        text = recognizer.recognize_google(audio)
    except sr.UnknownValueError:
        text = 'Parsing failed. Please try again.'

    # Don't just use capitalize because this removes existing capitals.
    # Probably don't have these anyway (transcription seems to usually be
    # lowercase) but just being safe here.
    text = text[0].upper() + text[1:]

    # Update windows now that transcription is complete.
    set_value(data['target_id'], text)
    for id_ in show_during:
        hide_item(id_)
    for id_ in data.get('show_after_ids', []):
        show_item(id_)

    # Manually call this so prompt is updated once we finish recording.
    task_select_callback('task_list',
                         data={'task_list_id': 'task_list',
                               'text_source_id': 'transcribed_text'})


def text_edit_callback(sender, data):
    """Triggered when user types in transcription text field. This way user
    edits update the prompt before making a query (this is often necessary
    since transcriptions are not always perfect).
    """
    task_select_callback('task_list',
                         data={'task_list_id': 'task_list',
                               'text_source_id': 'transcribed_text',
                               'update_kwargs': False})


def task_select_callback(sender, data):
    """data keys:
        - task_list_id (str: element containing selected item. Returns an int.)
        - text_source_id (str: element containing text for prompt input)
        - update_kwargs (bool: specifies whether to update query kwargs like
         max_tokens. We want those to be updated when we change the task but
             not when we manually call this function from our
             text_edit_callback.)
    """
    task_name, user_text = app.get_prompt_text(
        task_list_id=data['task_list_id'],
        text_source_id=data['text_source_id'],
        do_format=False
    )
    set_value('prompt', MANAGER.prompt(task_name, user_text))
    if not data.get('update_kwargs', True): return

    # Can't just use app.get_query_kwargs() because that merely retrieves what
    # the GUI currently shows. We want the default kwargs which are stored by
    # our prompt manager.
    kwargs = MANAGER.kwargs(task_name)
    kwargs.setdefault('stop', '')
    # Fixed value for max_tokens doesn't make sense for this task.
    if task_name == 'punctuate':
        kwargs['max_tokens'] = int(len(user_text.split()) * 2)
    for k, v in kwargs.items():
        # Choice of whether to mock calls is more related to the purpose of the
        # user session than the currently selected prompt.
        if k == 'mock':
            continue
        if k == 'stop' and isinstance(v, list):
            v = '\n'.join(v)
        set_value(k, v)


def query_callback(sender, data):
    """data keys:
        - target_id (str: element to display text response in)
        - interrupt_id (str: button to interrupt speaker if enabled)
    """
    show_item(data['query_msg_id'])
    kwargs = app.get_query_kwargs()
    # Can't pass empty list in for stop parameter.
    kwargs['stop'] = kwargs['stop'] or None
    task, text = app.get_prompt_text(do_format=False)
    model = MODEL_NAMES[get_value('model')]
    if 'neo' in model:
        kwargs.update(mock_func=query_gpt_neo, size=model.split()[-1],
                      mock=True)
    elif model == 'naive':
        kwargs.update(mock=True, mock_func=None)
    # eprint(select(kwargs, drop=['prompt']).items()) # TODO: rm
    try:
        _, res = MANAGER.query(task=task, text=text, **kwargs)
    except Exception as e:
        print(e)
        res = 'Query failed. Please check your settings and try again.'

    # GPT3 seems to like a type of apostrophe that dearpygui can't display.
    res = res.replace('’', "'")
    set_value(data['target_id'], res)
    hide_item(data['query_msg_id'])
    print('res', res)

    # Read response if desired. Threads allow us to interrupt speaker if user
    # checks a checkbox. This was surprisingly difficult - I settled on a
    # partial solution that can only quit after finishing saying a
    # sentence/line, so there may be a bit of a delayed response after asking
    # to interrupt.
    if get_value(data['read_checkbox_id']):
        show_item(data['interrupt_id'])
        errors = []
        thread = Thread(target=monitor_interrupt_checkbox,
                        args=(data['interrupt_id'], errors))
        thread.start()
        for chunk in sent_tokenize(res):
            SPEAKER.speak(chunk)
            if errors:
                set_value(data['interrupt_id'], False)
                break
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


def monitor_interrupt_checkbox(box_id, errors, wait=1, quit_after=None):
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
    """
    start = time.perf_counter()
    while True:
        if get_value(box_id):
            errors.append(True)
            print('Checkbox monitor quitting due to checkbox selection.')
            break
        time.sleep(wait)
        if not SPEAKER.is_speaking:
            print('Checkbox monitor quitting due to end of speech.')
            break
        if quit_after and time.perf_counter() - start > quit_after:
            print(f'Checkbox monitor quitting due to time exceeded.')
            break


def resize_callback(sender):
    width, height = get_main_window_size()
    # Resize callback is one of the few to not accept data. We have to find app
    # var in the global scope.
    app.recompute_dimensions(width, height)
    for i, id_ in enumerate(['input_window', 'options_window']):
        set_item_width(id_, app.widths[.5])
        set_item_height(id_, app.heights[1.])
        set_window_pos(id_, *app.pos[0][i])

        # 8 padding lengths seems to eliminate horizontal scrolling. Don't
        # fully understand why (doesn't quite match my calculations) but just
        # go with it.
        change_types = {'InputText', 'Listbox'}
        for child in get_item_children(id_):
            if get_item_type(child).split('::')[-1] in change_types:
                set_item_width(child, app.widths[.5] - 8*app.pad)

    # Images don't show up as children so we have to update them separately.
    img_size = _img_dims(get_item_width('conversation_img'),
                         get_item_height('conversation_img'),
                         width=app.widths[.5] - 8*app.pad)
    set_item_width('conversation_img', img_size['width'])
    set_item_height('conversation_img', img_size['height'])


class App:

    def __init__(self, width=1_200, height=760, font_size=22,
                 font_path='data/fonts/OpenSans-Light.ttf', theme='dark',
                 width_pcts=(.5, 1.), height_pcts=(.5, 1.), pad=5):
        self.width = width
        self.height = height
        self.pad = pad
        self.width_pcts = tolist(width_pcts)
        self.height_pcts = tolist(height_pcts)
        self.widths = {}
        self.heights = {}
        self.pos = []
        # These are populated in self.left_column().
        self.query_kwarg_ids = []
        self.recompute_dimensions(width, height)
        set_main_window_size(self.width, self.height)
        if font_path: add_additional_font(font_path, size=font_size)
        set_theme(theme.title())
        set_resize_callback(resize_callback)

    def _recompute_dimensions(self, dim, mode='width') -> dict:
        pcts = getattr(self, f'{mode}_pcts')
        # Relies on us using a quadrant grid layout. Left/top items need to
        # allow for 3 padding occurrences while right/bottom items only need to
        # account for 2.
        return {p: int((dim - ((3-i)*self.pad)) * p)
                for i, p in enumerate(pcts)}

    def recompute_dimensions(self, width, height):
        self.widths = self._recompute_dimensions(width, mode='width')
        self.heights = self._recompute_dimensions(height, mode='height')
        self.pos = self._recompute_positions()

    def _recompute_positions(self):
        # [0][0] is top left, [0][1] is top right, etc.
        pos = [
            [(self.pad, self.pad),
             (self.widths[.5] + 2*self.pad, self.pad)],
            [(self.pad, self.heights[.5] + 2*self.pad),
             (self.widths[.5] + 2*self.pad, self.heights[.5] + 2*self.pad)]
        ]
        return pos

    def get_prompt_text(self, task_list_id='task_list',
                        text_source_id='transcribed_text', do_format=True):
        """

        Parameters
        ----------
        task_list_id
        text_source_id
        do_format: bool
            If True, format prompt to get a single string with user input
            integrated. If False, return tuple of (task_name, user_input)
            which we can pass to manager.query().

        Returns
        -------
        str or tuple: str if do_format=True, tuple otherwise.
        """
        task_name = NAME2TASK[get_value(task_list_id)]
        input_ = get_value(text_source_id)

        # Auto-transcription doesn't add colon at end so we do it manually.
        if task_name == 'how_to' and input_ and not input_.endswith(':'):
            input_ += ':'
        set_value(text_source_id, input_)

        if do_format:
            return MANAGER.prompt(task_name, text=input_)
        return task_name, input_

    def get_query_kwargs(self):
        # Gets currently selected kwargs from GUI.
        kwargs = {name: get_value(name) for name in self.query_kwarg_ids}
        kwargs['stop'] = kwargs['stop'].splitlines()
        kwargs['prompt'] = self.get_prompt_text()
        return kwargs

    def menu(self):
        with window('menu_window'):
            with menu_bar('menu_bar'):
                with menu('Main'):
                    add_menu_item('Save')
                with menu('Settings'):
                    add_menu_item('Preferences')

    def primary_window(self):
        # Just here to suppress the default background and its un-settable
        # color. Do not delete.
        with window('primary_window'):
            pass

    def left_column(self):
        with window('input_window', width=self.widths[.5],
                    height=self.heights[1.] - self.pad, x_pos=self.pad,
                    y_pos=self.pad, no_resize=True, no_move=True):
            add_button('record_btn', label='Record',
                       callback_data={'show_during_ids': ['record_msg'],
                                      'target_id': 'transcribed_text'},
                       callback=transcribe_callback)
            with tooltip('record_btn', 'record_btn_tooltip'):
                add_text('Press and begin talking.\nSimply stop talking when '
                         'done and\nthe transcribed text should appear\n'
                         'within several seconds.')

            add_text('record_msg', default_value='Recording in progress...',
                     show=False)

            # Label is displayed next to input unless we manually suppress it.
            add_input_text('transcribed_text', default_value='',
                           multiline=True, width=self.widths[.5] - 8*self.pad,
                           height=300)
            set_item_label('transcribed_text', '')
            set_key_press_callback(text_edit_callback)
            add_spacing(count=3)
            add_text('Response')
            with tooltip('Response', 'Response_tooltip'):
                add_text('GPT3\'s response will be shown\nbelow after you hit '
                         'the\nQuery button.')

            add_text('query_progress_msg',
                     default_value='Query in progress...', show=False)
            add_checkbox('interrupt_checkbox', label='Interrupt', show=False)
            add_input_text('response_text', default_value='',
                           multiline=True, width=self.widths[.5] - 2*self.pad,
                           height=300)
            set_item_label('response_text', '')

            # We need to set initial dims before updating in resize_callback
            # otherwise we'll get a divide by zero error.
            img_path = most_recent_filepath('data/tmp')
            add_image('conversation_img', str(img_path),
                      **img_dims(img_path, width=self.widths[.5] - 8*self.pad))

    def right_column(self):
        with window('options_window', width=self.widths[.5],
                    height=self.heights[1.],
                    x_pos=self.widths[.5] + 2*self.pad,
                    y_pos=self.pad, no_resize=True, no_move=True):
            add_button('query_btn', label='Query', callback=query_callback,
                       callback_data={'target_id': 'response_text',
                                      'read_checkbox_id': 'read_response',
                                      'interrupt_id': 'interrupt_checkbox',
                                      'query_msg_id': 'query_progress_msg'})
            add_same_line()
            add_checkbox('read_response', label='read response',
                         default_value=True)
            with tooltip('read_response', 'read_response_tooltip'):
                add_text('Check this box if you want GPT3\'s response\n to be '
                         'read aloud.')

            add_radio_button('model', items=MODEL_NAMES)
            with tooltip('model', 'model_tooltip'):
                add_text('OpenAI\'s GPT3 produces the best results.\n'
                         'EleutherAI\'s GPT-Neo models are a solid free '
                         'alternative.\nNaive is mostly for debugging and '
                         'will load a saved\nresponse from GPT3 for the Dates '
                         'task\n(see below).')

            with label_above('task_list', 'Tasks:'):
                add_listbox('task_list', items=list(NAME2TASK),
                            num_items=len(NAME2TASK),
                            callback=task_select_callback,
                            callback_data={'task_list_id': 'task_list',
                                           'text_source_id': 'transcribed_text'})

            with tooltip('task_list', 'task_list_tooltip'):
                add_text('Select a task to perform.\nAll of these accept some '
                         'input\ntext and produce new text,\noften in response '
                         'to or\nbased on the input. Tasks\nprefixed by '
                         '"(debug)" are\nunaffected by input.')

            # Set default values for no specific task. These will be updated
            # once a task is selected.
            query_params = select(params(query_gpt3),
                                  drop=['prompt', 'stream', 'return_full',
                                        'strip_output', 'mock'])
            for k, v in query_params.items():
                v = v.default
                if isinstance(v, float):
                    add_input_float(k, default_value=v, min_value=0.0,
                                    max_value=1.0)
                    with tooltip(k, f'{k}_tooltip'):
                        add_text(
                            'Choose a value in [0.0, 1.0].\nHigher values '
                            'produce more whimsical\ntext while lower values '
                            'are more\nstraightforward/businesslike.'
                        )
                elif isinstance(v, int):
                    if k == 'engine_i':
                        min_value, max_value = 0, 3
                        tooltip_text = 'There are 4 engines, indexed\nfrom ' \
                                       'zero, where larger numbers\n' \
                                       'correspond to more powerful models\n' \
                                       'and better outputs.'
                    else:
                        min_value = 1
                        input_toks = get_value('transcribed_text').split()
                        # Compute n_tokens conservatively to be safe.
                        max_value = 2048 - int(1.67 * len(input_toks))
                        tooltip_text = 'Specifies the max number of\ntokens ' \
                                       'in the output. GPT3\ncan process a ' \
                                       'total of 2048\ntokens at once, ' \
                                       'counting the input.\nGPT-Neo can ' \
                                       'only produce up\nto 250 token ' \
                                       'outputs (not counting\ninputs). ' \
                                       'Note that a word is ~1.33\ntokens ' \
                                       'on average.'
                    add_input_int(k, default_value=v, min_value=min_value,
                                  max_value=max_value)
                    with tooltip(k, f'{k}_tooltip'):
                        add_text(tooltip_text)
                else:
                    continue
                self.query_kwarg_ids.append(k)

            add_spacing(count=2)
            with label_above('stop', 'Stop phrases:'):
                add_input_text('stop', default_value='', hint='TODO: hint',
                               multiline=True, height=90)
            with tooltip('stop', 'stop_tooltip'):
                add_text('You can enter one or more phrases\nwhich will '
                         'signal for the model to\nstop generating text if it '
                         'encounters\nthem. These "stop phrases" will be\n'
                         'excluded from the final output.\nEach phrase must '
                         'be on its own line in\nthis text box')
            self.query_kwarg_ids.append('stop')

            # I'd like to avoid the horizontal scrollbar but there's no
            # straightforward way to do this in dearpygui at the moment.
            add_spacing(count=2)
            with label_above('prompt', 'Prompt:'):
                add_input_text('prompt', default_value=self.get_prompt_text(),
                               multiline=True, width=self.widths[.5] - 2*self.pad,
                               height=450)
            with tooltip('prompt', 'prompt_tooltip'):
                add_text('This is the full prompt that will\nbe sent to '
                         'GPT3. If you haven\'t\nrecorded or typed anything '
                         'in\nthe input box yet, you may\nsee a pair of curly '
                         'braces in here:\nthis will be replaced by\nyour '
                         'input once you provide it.')

    def build(self):
        self.primary_window()
        # self.menu()
        self.left_column()
        self.right_column()

    def run(self):
        self.build()
        start_dearpygui(primary_window='primary_window')


if __name__ == '__main__':
    app = App()
    app.run()
