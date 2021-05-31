from dearpygui.core import *
from dearpygui.simple import *
import os
import speech_recognition as sr
import time
from threading import Thread

from htools.core import tolist, select, eprint, flatten, hsplit
from htools.meta import params
from htools.structures import IndexedDict
from jabberwocky.openai_utils import PromptManager, query_gpt3, query_gpt_neo
from jabberwocky.speech import Speaker


os.chdir('../')
MANAGER = PromptManager(verbose=False)
NAME2TASK = IndexedDict({
    'Punctuate': 'punctuate',
    'Summarize': 'tldr',
    'Explain Like I\'m 5': 'eli',
    'Explain Machine Learning': 'simplify_ml',
    'Machine Learning Abstract Writer': 'ml_abstract',
    'How To': 'how_to',
    'Dates (debug)': 'short_dates',
    'Math (debug)': 'shortest'
})
MODEL_NAMES = ['gpt3', 'gpt-neo 2.7B', 'gpt-neo 1.3B', 'gpt-neo 125M', 'naive']
SPEAKER = Speaker(newline_pause=400)


def transcribe_callback(sender, data):
    """data keys:
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
    task_select_callback('task_list',
                         data={'task_list_id': 'task_list',
                               'text_source_id': 'transcribed_text'})


def punctuate_callback(sender, data):
    """data keys:
        - source (str: element containing text to punctuate)
    """
    text = get_value(data['source'])
    set_value(data['source'], text.swapcase())


def task_select_callback(sender, data):
    """data keys:
        - task_list_id (str: element containing selected item. Returns an int.)
        - text_source_id (str: element containing text for prompt input)
    """
    task_name, user_text = app.get_prompt_text(
        task_list_id=data['task_list_id'],
        text_source_id=data['text_source_id'],
        do_format=False
    )
    set_value('prompt', MANAGER.prompt(task_name, user_text))

    # Can't just use app.get_query_kwargs() because that merely retrieves what
    # the GUI currently shows. We want the default kwargs which are stored by
    # our prompt manager.
    kwargs = MANAGER.kwargs(task_name)
    kwargs.setdefault('stop', '')
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
    eprint(select(kwargs, drop=['prompt']).items())
    try:
        _, res = MANAGER.query(task=task, text=text, **kwargs)
    except Exception as e:
        print(e)
        res = 'Query failed. Please check your settings and try again.'
    set_value(data['target_id'], res)

    # Read response if desired. Threads allow us to interrupt speaker if user
    # checks a checkbox. This was surprisingly difficult - I settled on a
    # partial solution that can only quit after finishing saying a
    # sentence/line, so there may be a bit of a delayed response after asking
    # to interrupt.
    if get_value(data['read_checkbox_id']):
        show_item(data['interrupt_id'])
        errors = []
        thread = Thread(target=monitor_interrupt_btn,
                        args=(data['interrupt_id'], errors))
        thread.start()
        for chunk in flatten(hsplit(line, '.') for line in res.splitlines()):
            SPEAKER.speak(chunk)
            if errors:
                set_value(data['interrupt_id'], False)
                break
        hide_item(data['interrupt_id'])
        thread.join()


# TODO start
def monitor_speaker(speaker, name, wait=2, quit_after=15):
    start = time.perf_counter()
    while True:
        print(f'[{name}] speaking: ' + str(speaker.is_speaking))
        time.sleep(wait)
        if time.perf_counter() - start > quit_after:
            print(f'[{name}]: quitting due to time exceeded')
            break


def monitor_interrupt_btn(btn_id, errors, quit_after=15):
    start = time.perf_counter()
    while True:
        if get_value(btn_id):
            errors.append(True)
            break
        time.sleep(1)
        if time.perf_counter() - start > quit_after:
            print(f'btn monitor quitting due to time exceeded')
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
        input_text = get_value(text_source_id)
        if do_format:
            return MANAGER.prompt(task_name, text=input_text)
        return task_name, input_text

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
                                      'target_id': 'transcribed_text',
                                      'show_after_ids': ['punctuate_btn',
                                                         'punctuate_tooltip']},
                       callback=transcribe_callback)
            add_text('record_msg', default_value='Recording in progress...',
                     show=False)

            # Label is displayed next to input unless we manually suppress it.
            add_input_text('transcribed_text', default_value='',
                           multiline=True, width=self.widths[.5] - 2*self.pad,
                           height=300)
            set_item_label('transcribed_text', '')
            set_key_press_callback(text_edit_callback)

            # Button only appears when there is text to punctuate.
            add_button('punctuate_btn', label='Auto-Punctuate',
                       show=False,
                       callback_data={'source': 'transcribed_text',
                                      'target': 'transcribed_text'},
                       callback=punctuate_callback)

            # TODO: add_tooltip produces "SystemError: returned a result with
            # an error set.
            with tooltip('punctuate_btn', 'punctuate_tooltip', show=False):
                add_text('Auto-punctuate input with GPT3. You may also choose '
                         'to clean up the text manually.')

            add_spacing(count=3)
            add_text('Response')
            add_same_line()
            add_checkbox('interrupt_checkbox', label='Interrupt', show=False)
            add_input_text('response_text', default_value='',
                           multiline=True, width=self.widths[.5] - 2*self.pad,
                           height=300)
            set_item_label('response_text', '')

    def right_column(self):
        with window('options_window', width=self.widths[.5],
                    height=self.heights[1.],
                    x_pos=self.widths[.5] + 2*self.pad,
                    y_pos=self.pad, no_resize=True, no_move=True):
            add_button('query_btn', label='Query', callback=query_callback,
                       callback_data={'target_id': 'response_text',
                                      'read_checkbox_id': 'read_response',
                                      'interrupt_id': 'interrupt_checkbox'})
            add_same_line()
            add_checkbox('read_response', label='read response',
                         default_value=True)
            add_radio_button('model', items=MODEL_NAMES)
            add_listbox('task_list', items=list(NAME2TASK),
                        num_items=len(NAME2TASK),
                        callback=task_select_callback,
                        callback_data={'task_list_id': 'task_list',
                                       'text_source_id': 'transcribed_text'})

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
                elif isinstance(v, int):
                    add_input_int(k, default_value=v,
                                  min_value=0 if k == 'engine_i' else 1,
                                  max_value=3 if k == 'engine_i' else 256)
                else:
                    print('NOT DISPLAYED', k, v)
                    continue
                self.query_kwarg_ids.append(k)

            add_spacing(count=2)
            add_input_text('stop', default_value='', hint='TODO: hint',
                           multiline=True)
            self.query_kwarg_ids.append('stop')

            # I'd like to avoid the horizontal scrollbar but there's no
            # straightforward way to do this in dearpygui at the moment.
            # add_label_text('test label') # TODO rm
            add_spacing(count=2)
            add_input_text('prompt', default_value=self.get_prompt_text(),
                           multiline=True, width=self.widths[.5] - 2*self.pad,
                           height=450)

    def build(self):
        self.primary_window()
        # self.menu()
        self.left_column()
        self.right_column()

    def run(self):
        self.build()
        start_dearpygui(primary_window='primary_window')


if __name__ == '__main__':
    MONITOR = Thread(target=monitor_speaker, args=(SPEAKER, 'MAIN_THREAD'))
    MONITOR.start()

    app = App()
    app.run()

    MONITOR.join()

