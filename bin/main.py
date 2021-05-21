from dearpygui.core import *
from dearpygui.simple import *
import os
import speech_recognition as sr

from htools.core import tolist, select, eprint
from htools.meta import params
from htools.structures import IndexedDict
from jabberwocky.openai_utils import PromptManager, query_gpt3


os.chdir('../')
MANAGER = PromptManager(verbose=False)
NAME2TASK = IndexedDict({
    'Punctuate': 'punctuate',
    'Summarize': 'tldr',
    'Explain Like I\'m 5': 'eli',
    'Explain Machine Learning': 'simplify_ml',
    'How To': 'how_to',
    'Dates (debug)': 'short_dates',
    'Math (debug)': 'shortest'
})


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
    for k, v in MANAGER.kwargs(task_name).items():
        print(k, v)
        set_value(k, v)


def query_callback(sender, data):
    """data keys:
        - target_id (str: element to display text response in)
    """
    kwargs = app.get_query_kwargs()
    eprint(kwargs.items())
    task, text = app.get_prompt_text(do_format=False)
    _, res = MANAGER.query(task=task, text=text, **kwargs)
    set_value(data['target_id'], res)


def resize_callback(sender):
    width, height = get_main_window_size()
    # Resize callback is one of the few to not accept data. We have to find app
    # var in the global scope.
    app.recompute_dimensions(width, height)
    for i, id_ in enumerate(['input_window', 'options_window',
                             'output_window']):
        set_item_width(id_, app.widths[.5])
        set_item_height(id_, app.heights[.5])
        set_window_pos(id_, *app.pos[i % 2][i // 2])


class App:

    def __init__(self, width=1_200, height=760, font_scale=1.25, theme='dark',
                 width_pcts=(.5, 1.), height_pcts=(.5, 1.), pad=5):
        self.width = width
        self.height = height
        self.pad = pad
        self.width_pcts = tolist(width_pcts)
        self.height_pcts = tolist(height_pcts)
        self.widths = {}
        self.heights = {}
        self.pos = []
        # These are populated in self.input_column().
        self.query_kwarg_ids = ['mock']
        self.recompute_dimensions(width, height)
        set_main_window_size(self.width, self.height)
        set_global_font_scale(font_scale)
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

    def input_column(self):
        with window('input_window', width=self.widths[.5],
                    height=self.heights[.5] - self.pad, x_pos=self.pad,
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

        with window('options_window', width=self.widths[.5],
                    height=self.heights[.5] // 2, x_pos=self.pad, no_move=True,
                    y_pos=self.pad + self.heights[.5], no_resize=True):
            add_button('query_btn', label='Query', callback=query_callback,
                       callback_data={'target_id': 'response_text'})
            add_same_line()
            add_checkbox('mock')
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
                # if isinstance(v, bool):
                #     add_checkbox(k, default_value=v)
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

            # I'd like to avoid the horizontal scrollbar but there's no
            # straightforward way to do this in dearpygui at the moment.
            # prompt = MANAGER.prompt(NAME2TASK[get_value('task_list')],
            #                         text=get_value('transcribed_text'))
            add_input_text('prompt', default_value=self.get_prompt_text(),
                           multiline=True, width=self.widths[.5] - 2*self.pad,
                           height=300)

    def output_column(self):
        with window('output_window', width=self.widths[.5],
                    height=self.heights[1.],
                    x_pos=self.widths[.5] + 2*self.pad,
                    y_pos=self.pad, no_resize=True, no_move=True):
            add_button('output button')
            add_input_text('response_text', default_value='',
                           multiline=True, width=self.widths[.5] - 2*self.pad,
                           height=300)
            set_item_label('response_text', '')

    def build(self):
        self.primary_window()
        # self.menu()
        self.input_column()
        self.output_column()

    def run(self):
        self.build()
        start_dearpygui(primary_window='primary_window')


if __name__ == '__main__':
    app = App()
    app.run()
