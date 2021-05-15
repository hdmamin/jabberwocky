from dearpygui.core import *
from dearpygui.simple import *
import speech_recognition as sr


def transcribe(sender, data):
    msg_id = data.pop('message_target', '')
    if msg_id: show_item(msg_id)

    # Record until pause.
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print('in mic')
        audio = recognizer.listen(source)
    try:
        print('in parsing try')
        text = recognizer.recognize_google(audio)
    except sr.UnknownValueError:
        print('Failed to parse')
        text = 'Parsing failed. Please try again.'
    set_value(data['target'], text)
    # set_value(msg_id, '')
    hide_item(msg_id)


def punctuate(sender, data):
    text = get_value(data['source'])
    set_value(data['source'], text.swapcase())


class App:

    def __init__(self, width=900, height=720, font_scale=1.25,
                 theme='Classic', width_pcts=[.5, .5], pad=5):
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

    def input_window(self):
        with window('input_window', width=self.widths[0],
                    height=self.heights[0]):
            set_window_pos('input_window', x=self.pad, y=self.pad)
            add_button('record_btn',
                       callback_data={'message_target': 'record_msg',
                                      'target': 'transcribed_text'},
                       callback=transcribe)
            add_text('record_msg', default_value='Recording in progress...',
                     show=False)
            # add_text('transcribed_text')
            add_input_text('transcribed_text', height=100)
            # add_input_text('input text', default_value='abc')
            # add_slider_float('number', default_value=.5, max_value=10)
            add_button('punctuate_btn',
                       callback_data={'source': 'transcribed_text',
                                      'target': 'transcribed_text'},
                       callback=punctuate)

    def output_window(self):
        with window('output_window', width=self.widths[1],
                    height=self.heights[1]):
            set_window_pos('output_window', x=self.widths[0] + 2*self.pad,
                           y=self.pad)
            add_button('output button')

    def build(self):
        self.input_window()
        self.output_window()

    def run(self):
        self.build()
        start_dearpygui()


if __name__ == '__main__':
    App().run()
