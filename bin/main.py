from dearpygui.core import *
from dearpygui.simple import *
import os
import speech_recognition as sr

from htools.core import tolist
from htools.meta import params
from htools.structures import IndexedDict
from jabberwocky.openai_utils import PromptManager, query_gpt3


os.chdir('../')
MANAGER = PromptManager(verbose=False)

# TASK2NAME = {
#     'punctuate': 'Punctuate',
#     'tldr': 'Summarize',
#     'eli': 'Explain Like I\'m 5',
#     'simplify_ml': 'Explain Machine Learning',
#     'how_to': 'How To',
#     'short_dates': 'Dates (debug)',
#     'shortest': 'Math (debug)'
# }
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


def punctuate_callback(sender, data):
    text = get_value(data['source'])
    set_value(data['source'], text.swapcase())


def task_select_callback(sender, data):
    task_name = NAME2TASK[get_value(sender)]
    for k, v in MANAGER.kwargs(task_name).items():
        set_value(k, v)


def resize_callback(sender):
    width, height = get_main_window_size()
    # Resize callback is one of the few to not accept data. We have to find app
    # var in the global scope.
    app.recompute_dimensions(width, height)
    print(app.widths)
    for i, id_ in enumerate(['input_window', 'options_window',
                             'output_window']):
        set_item_width(id_, app.widths[.5])
        set_item_height(id_, app.heights[.5])
        set_window_pos(id_, *app.pos[i % 2][i // 2])


class App:

    def __init__(self, width=900, height=720, font_scale=1.25, theme='dark',
                 width_pcts=(.5, 1.), height_pcts=(.5, 1.), pad=5):
        self.width = width
        self.height = height
        self.pad = pad
        self.width_pcts = tolist(width_pcts)
        self.height_pcts = tolist(height_pcts)
        self.widths = {}
        self.heights = {}
        self.pos = []
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
                    y_pos=self.pad):
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
                    height=self.heights[.5] // 2, x_pos=self.pad,
                    y_pos=self.pad + self.heights[.5]):
            add_listbox('task_list', items=list(NAME2TASK),
                        num_items=len(NAME2TASK),
                        callback=task_select_callback,
                        callback_data={'parent_id': 'options_window'})
            print('in window')
            for k, v in params(query_gpt3).items():
                v = v.default
                # TODO: put prompt last after other options.
                # TODO: format so no long horizontal scrollbar.
                # TODO: update callback to update value when prompt changes.
                if k == 'prompt':
                    task_i = get_value('task_list')
                    print(task_i)
                    prompt = MANAGER.prompt(NAME2TASK[task_i])
                    add_text(k, default_value=prompt)
                elif isinstance(v, bool):
                    add_checkbox(k, default_value=v)
                elif isinstance(v, float):
                    add_slider_float(k, default_value=v, min_value=0.0,
                                     max_value=1.0)
                elif isinstance(v, int):
                    add_input_int(k, default_value=v)
                else:
                    print('NOT DISPLAYED', k, v)

    def output_column(self):
        with window('output_window', width=self.widths[.5],
                    height=self.heights[1.],
                    x_pos=self.widths[.5] + 2*self.pad,
                    y_pos=self.pad):
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
