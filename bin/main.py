from dearpygui.core import *
from dearpygui.simple import *
import speech_recognition as sr


def transcribe(sender, data):
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


def punctuate(sender, data):
    text = get_value(data['source'])
    set_value(data['source'], text.swapcase())


class App:

    def __init__(self, width=900, height=720, font_scale=1.25,
                 theme='dark', width_pcts=[.5, .5], pad=5):
        self.width = width
        self.height = height
        self.pad = pad
        self.width_pcts = width_pcts
        self.widths = self._compute_widths()
        self.heights = [height - 2*pad] * len(width_pcts)
        set_main_window_size(self.width, self.height)
        set_global_font_scale(font_scale)
        set_theme(theme.title())

    def _compute_widths(self):
        n = len(self.width_pcts)
        width = self.width - (n + 1)*self.pad
        return [int(width * p) for p in self.width_pcts]

    def menu(self):
        with window('menu_window'):
            with menu_bar('menu_bar'):
                with menu('Main'):
                    add_menu_item('Save')
                with menu('Settings'):
                    add_menu_item('Preferences')

    def primary_window(self):
        with window('primary_window'):
            pass

    def input_window(self):
        with window('input_window', width=self.widths[0],
                    height=self.heights[0]):
            set_window_pos('input_window', x=self.pad, y=self.pad)
            add_button('record_btn', label='Record',
                       callback_data={'show_during_ids': ['record_msg'],
                                      'target_id': 'transcribed_text',
                                      'show_after_ids': ['punctuate_btn',
                                                         'punctuate_tooltip']},
                       callback=transcribe)
            add_text('record_msg', default_value='Recording in progress...',
                     show=False)

            # Label is displayed next to input unless we manually suppress it.
            add_input_text('transcribed_text', default_value='',
                           multiline=True, width=250, height=300)
            set_item_label('transcribed_text', '')

            # Button only appears when there is text to punctuate.
            add_button('punctuate_btn', label='Auto-Punctuate',
                       show=False,
                       callback_data={'source': 'transcribed_text',
                                      'target': 'transcribed_text'},
                       callback=punctuate)

            # TODO: add_tooltip produces "SystemError: returned a result with
            # an error set.
            with tooltip('punctuate_btn', 'punctuate_tooltip', show=False):
                add_text('Auto-punctuate input with GPT3. You may also choose '
                         'to clean up the text manually.')
            # add_slider_float('number', default_value=.5, max_value=10)

    def output_window(self):
        with window('output_window', width=self.widths[1],
                    height=self.heights[1]):
            set_window_pos('output_window', x=self.widths[0] + 2*self.pad,
                           y=self.pad)
            add_button('output button')

    def build(self):
        self.primary_window()
        # self.menu()
        self.input_window()
        self.output_window()

    def run(self):
        self.build()
        start_dearpygui(primary_window='primary_window')


if __name__ == '__main__':
    App().run()
