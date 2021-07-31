"""Launches a GUI providing an audio interface to GPT3.

Example usage:
--------------
python bin/main.py
"""

# I think dearpygui imports a different contextmanager so we rename this one.
from dearpygui.core import *
from dearpygui.simple import *
import os

from htools.core import tolist, select
from htools.meta import params
from htools.structures import IndexedDict
from jabberwocky.core import GuiTextChunker
from jabberwocky.openai_utils import PromptManager, ConversationManager, \
    query_gpt3
from jabberwocky.speech import Speaker
from jabberwocky.utils import set_module_globals

from utils import label_above
from callbacks import *


os.chdir('../')


class App:
    """Dearpygui app. Was trying to organize things instead of having 1 giant
    glob of context managers but I'm not sure this ended up any clearer. OOP
    examples were sparse for dearpygui since the library doesn't formally
    provide an API for that yet but I think I saw one sample that defined a
    new class for each window rather than a method for each window like I did,
    so that might be a better option in the future.
    """

    def __init__(self, width=1_200, height=760, font_size=22,
                 font_path='data/fonts/OpenSans-Light.ttf', theme='dark',
                 width_pcts=(.5, 1.), height_pcts=(.5, 1.), pad=5):
        """
        Parameters
        ----------
        width: int
            App main window size.
        height: int
            App main window height.
        font_size: int
            Font size of all text in app.
        font_path: str or Path
            Used to locate a local tff file defining a custom font.
        theme: str
            Defines dearpygui color scheme.
        width_pcts: Iterable[float]
            Defines possible column widths we might use: 0.5 means a column
            taking up half of the total width. Safest just to leave as defaults
            here.
        height_pcts: Iterable[float]
            Defines possible column widths we might use: 0.5 means a row takes
            up half the total height. Safest just to leave as defaults here.
        pad: int
            Size of padding to use between windows. Used to space out other
            elements too. I found dearpygui's formatting a bit confusing to
            control so this is probably also best left as default value.
        """
        self.width = width
        self.height = height
        self.pad = pad
        self.width_pcts = tolist(width_pcts)
        self.height_pcts = tolist(height_pcts)
        self.widths = {}
        self.heights = {}
        self.pos = []
        self.menu_height = 25
        # These are populated in self.left_column().
        self.query_kwarg_ids = []
        self.recompute_dimensions(width, height)
        set_main_window_size(self.width, self.height)
        if font_path: add_additional_font(font_path, size=font_size)
        set_theme(theme.title())
        set_resize_callback(resize_callback)

    def _recompute_dimensions(self, dim, mode='width') -> dict:
        """Helper when computing new dimensions when app window is resized."""
        pcts = getattr(self, f'{mode}_pcts')
        # Relies on us using a quadrant grid layout. Left/top items need to
        # allow for 3 padding occurrences while right/bottom items only need to
        # account for 2. Height computations are slightly different due to
        # global menu bar.
        pct2size = {}
        is_height = mode == 'height'
        for i, p in enumerate(pcts):
            size = dim - ((3-i)*self.pad) - self.menu_height*is_height
            pct2size[p] = int(size * p)
        return pct2size

    def recompute_dimensions(self, width, height):
        """Recompute new dimensions and positions of app elements when window
        is resized.

        Parameters
        ----------
        width: int
            Global window width.
        height: int
            Global window height.
        """
        self.widths = self._recompute_dimensions(width, mode='width')
        self.heights = self._recompute_dimensions(height, mode='height')
        self.pos = self._recompute_positions()

    def _recompute_positions(self):
        """Recompute new x,y positions for each column in our layout when the
        main window is resized.

        Returns
        -------
        list[list[tuple]]: Each tuple is a pair of (x, y) coordinates.
        pos[0][0] is top left, pos[0][1] is top right, pos[1][0] is bottom
        left, and pos[1][1] is bottom right.
        """
        pos = [
            [(self.pad, self.pad + self.menu_height),
             (self.widths[.5] + 2*self.pad, self.pad + self.menu_height)],
            [(self.pad, self.heights[.5] + 2*self.pad),
             (self.widths[.5] + 2*self.pad, self.heights[.5] + 2*self.pad)]
        ]
        return pos

    def get_prompt_text(self, task_list_id='task_list',
                        text_source_id='transcribed_text', do_format=True):
        """In default mode, get the current prompt. This can either be fully
        resolved or not, depending on your choice of do_format.

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
        if do_format:
            return MANAGER.prompt(task_name, text=input_)
        return task_name, input_

    def get_query_kwargs(self) -> dict:
        """Gets currently selected kwargs from GUI when in default mode. The
        user can set these via various widgets and we need to access them to
        make queries with the appropriate kwargs.
        """
        # Gets currently selected kwargs from GUI.
        kwargs = {name: get_value(name) for name in self.query_kwarg_ids}
        kwargs['stop'] = kwargs['stop'].splitlines()
        kwargs['prompt'] = self.get_prompt_text()
        return kwargs

    def primary_window(self):
        """Defines the main window which hours both column windows. This lets
        us override the default background color which can't be changed
        otherwise. This method also defines the menu bar which lets us switch
        between default and conv modes.
        """
        # Just here to suppress the default background and its un-settable
        # color. Do not delete.
        with window('primary_window'):
            with menu_bar('menu_bar'):
                with menu('Settings'):
                    with menu('Mode'):
                        add_menu_item('default_menu_choice',
                                      label='Default\t[x]',
                                      callback=menu_default_callback)
                        add_menu_item('conv_menu_choice', label='Conversation',
                                      callback=menu_conversation_callback)
        set_main_window_title('Jabberwocky')

    def left_column(self):
        """Defines the dearpygui elements in the left column of the app.
        Notice this includes both the default window and the conv window, only
        one of which is visible at a time. I believe I tried to separate them
        but encountered issues since dearpygui does some black magic with the
        current stack when defining windows.
        """
        with window('Input', width=self.widths[.5],
                    height=self.heights[1.], x_pos=self.pad,
                    y_pos=self.pad, no_resize=True, no_move=True):
            ###################################################################
            # Default window (1 time queries)
            ###################################################################
            add_button('record_btn', label='Record',
                       callback_data={'show_during_ids': ['record_msg'],
                                      'target_id': 'transcribed_text'},
                       callback=transcribe_callback)
            with tooltip('record_btn', 'record_btn_tooltip'):
                add_text('Press and begin talking.\nSimply stop talking when '
                         'done and\nthe transcribed text should appear\n'
                         'within several seconds.')
            add_same_line()
            add_button('autoformat_btn', label='Auto-Format',
                       callback=format_text_callback,
                       callback_data={'text_source_id': 'transcribed_text',
                                      'key': 'transcribed',
                                      'task_list_id': 'task_list'})
            with tooltip('autoformat_btn', 'autoformat_btn_tooltip'):
                add_text(
                    'This mostly inserts newlines so your text is all \n'
                    'visible without horizontal scrolling. Newlines that\n'
                    'you manually type will be considered a part of the\n'
                    'prompt and will be included in the text sent to GPT3,\n'
                    'so use that with intention. Just type your text as you\n'
                    'want the model to see it and use this Auto-Format\n'
                    'button for changes where the target viewer is you or\n'
                    'another human. This is done automatically for you when\n'
                    'using voice transcription, but if you type the input\n'
                    'or manually edit the transcription, you will need to\n'
                    'manually press the button.'
                )
            add_same_line()
            add_button('default_saveas_btn', label='Save As',
                       callback=saveas_callback,
                       callback_data={'dir_id': 'default_save_dir_text',
                                      'file_id': 'default_save_file_text',
                                      'task_list_id': 'task_list'})
            with tooltip('default_saveas_btn', 'default_saveas_btn_tooltip'):
                add_text('Do not save while speaking is in progress.')

            with popup('default_saveas_btn', 'Save Completion', modal=True,
                       width=450, mousebutton=mvMouseButton_Left):
                # Input dir and file names both get updated in save as callback
                # so the values here don't really matter.
                add_input_text('default_save_dir_text', label='Directory',
                               default_value='')
                add_input_text('default_save_file_text', label='File Name',
                               default_value='')
                add_same_line()
                add_checkbox('default_force_save_box', label='Force Save',
                             default_value=False)
                # Notice this can skip a few of the keys we need to provide in
                # conv mode.
                save_callback_data = {
                    'popup_id': 'Save Completion',
                    'error_msg_id': 'default_save_error_msg',
                    'dir_id': 'default_save_dir_text',
                    'file_id': 'default_save_file_text',
                    'force_save_id': 'default_force_save_box',
                    'task_list_id': 'task_list'
                }
                add_button('default_save_btn', label='Save',
                           callback=save_callback,
                           callback_data=save_callback_data)
                add_same_line()
                add_button('default_cancel_save_btn', label='Cancel',
                           callback=cancel_save_conversation_callback,
                           callback_data=save_callback_data)
                add_same_line()
                add_text('default_save_error_msg',
                         default_value='Failed to save file.', show=False)

            add_text('record_msg', default_value='Recording in progress...',
                     show=False)

            # Label is displayed next to input unless we manually suppress it.
            with label_above('transcribed_text'):
                add_input_text('transcribed_text', default_value='',
                               multiline=True,
                               width=self.widths[.5] - 8*self.pad, height=300)
            set_key_press_callback(text_edit_callback)
            add_spacing(count=2)
            add_text('Response')
            with tooltip('Response', 'Response_tooltip'):
                add_text('GPT3\'s response will be shown\nbelow after you hit '
                         'the\nQuery button.')

            add_text('query_progress_msg',
                     default_value='Query in progress...', show=False)
            add_checkbox('interrupt_checkbox', label='Interrupt', show=False)
            with label_above('response_text'):
                add_input_text('response_text', default_value='',
                               multiline=True,
                               width=self.widths[.5] - 2*self.pad, height=300)

        #######################################################################
        # Conversation Window
        #######################################################################
        with window('Conversation', width=self.widths[.5],
                    height=self.heights[1.], x_pos=self.pad,
                    y_pos=self.pad, no_resize=True, no_move=True, show=False):
            # Can't use persona_select_callback to do this because it calls
            # update_persona_info() which uses items that aren't defined until
            # we render right-hand window, and dearpygui complains.
            name = CONV_MANAGER.personas()[0]
            CONV_MANAGER.start_conversation(name, True)
            SPEAKER.voice = GENDER2VOICE[CONV_MANAGER.current_gender]
            set_key_press_callback(text_edit_callback)

            # Same as in default window but with different names/callback_data.
            add_button('conv_record_btn', label='Record',
                       callback_data={'show_during_ids': ['conv_record_msg'],
                                      'target_id': 'conv_text'},
                       callback=transcribe_callback)
            with tooltip('conv_record_btn', 'conv_record_btn_tooltip'):
                add_text('Press and begin talking.\nSimply stop talking when '
                         'done and\nthe transcribed text should appear\n'
                         'within several seconds.')
            add_same_line()
            add_button('conv_saveas_btn', label='Save As',
                       callback=saveas_callback,
                       callback_data={'dir_id': 'save_dir_text',
                                      'file_id': 'save_file_text'})
            with tooltip('conv_saveas_btn', 'conv_saveas_btn_tooltip'):
                add_text('Do not save while speaking is in progress.')
            with popup('conv_saveas_btn', 'Save Conversation', modal=True,
                       width=450, mousebutton=mvMouseButton_Left):
                # Input dir and file names both get updated in save as callback
                # so the values here don't really matter.
                add_input_text(
                    'save_dir_text', label='Directory',
                    default_value=str(CONV_MANAGER.conversation_dir.absolute())
                )
                add_input_text(
                    'save_file_text', label='File Name',
                    default_value=f'{CONV_MANAGER.current_persona}.txt'
                )
                add_checkbox('end_conv_box',
                             label='End Conversation', default_value=True)
                add_same_line()
                add_checkbox('force_save_box', label='Force Save',
                             default_value=False)
                add_button('conv_save_btn', label='Save',
                           callback=save_callback,
                           callback_data={'source_text_id': 'conv_text',
                                          'popup_id': 'Save Conversation',
                                          'error_msg_id': 'save_error_msg',
                                          'dir_id': 'save_dir_text',
                                          'file_id': 'save_file_text',
                                          'force_save_id': 'force_save_box',
                                          'end_conv_id': 'end_conv_box'})
                add_same_line()
                add_button('conv_cancel_save_btn', label='Cancel',
                           callback=cancel_save_conversation_callback,
                           callback_data={'popup_id': 'Save Conversation',
                                          'source_text_id': 'conv_text',
                                          'error_msg_id': 'save_error_msg',
                                          'dir_id': 'save_dir_text',
                                          'file_id': 'save_file_text',
                                          'force_save_id': 'force_save_box',
                                          'end_conv_id': 'end_conv_box'})
                add_same_line()
                add_text('save_error_msg',
                         default_value='Failed to save file.', show=False)
            add_same_line()

            # Other non-save-related buttons.
            add_button('end_conv_btn', label='End Conversation',
                       callback=end_conversation_callback)
            with tooltip('end_conv_btn', 'end_conv_btn_callback'):
                add_text('This will delete the current conversation. \nYou '
                         'will no longer be able to save a transcript.')
            add_text('conv_record_msg',
                     default_value='Recording in progress...',
                     show=False)

            # Visible when querying and speaking, respectively.
            add_text('conv_query_progress_msg',
                     default_value='Query in progress...', show=False)
            add_checkbox('conv_interrupt_checkbox', label='Interrupt',
                         show=False)
            add_text('edit_warning_msg', show=False,
                     default_value='You can only edit your most recent '
                                   'speaking turn.')

            # Just tweaked height until it seemed to do what I want (no
            # vertical scroll w/ default window size). Not sure how to
            # calculate precisely what I want (unknown height of query button).
            add_input_text('conv_text', label='', default_value='',
                           multiline=True, width=self.widths[.5] - 8*self.pad,
                           height=self.heights[1] - 16*self.pad)

    def right_column(self):
        """Defines the dearpygui elements in the right column of the app.
        Notice this includes both the default window and the conv window, only
        one of which is visible at a time. I believe I tried to separate them
        but encountered issues since dearpygui does some black magic with the
        current stack when defining windows.
        """
        with window('Options', width=self.widths[.5],
                    height=self.heights[1.],
                    x_pos=self.widths[.5] + 2*self.pad,
                    y_pos=self.pad, no_resize=True, no_move=True):

            ###################################################################
            # Default Options Window
            ###################################################################
            add_button('query_btn', label='Query', callback=query_callback,
                       callback_data={'target_id': 'response_text',
                                      'read_checkbox_id': 'read_response',
                                      'interrupt_id': 'interrupt_checkbox',
                                      'query_msg_id': 'query_progress_msg'})
            add_same_line()
            add_checkbox('read_response', label='Read Response',
                         default_value=True)
            with tooltip('read_response', 'read_response_tooltip'):
                add_text('Check this box if you want GPT3\'s response\n to be '
                         'read aloud.')
            add_same_line()
            add_input_int('default_speed_input', default_value=0, min_value=0,
                          max_value=10, min_clamped=True, max_clamped=True,
                          label='Speaker Speed',
                          width=int(APP.widths[.5] * .35),
                          callback=speaker_speed_callback)

            add_radio_button('model', items=MODEL_NAMES)
            with tooltip('model', 'model_tooltip'):
                add_text('OpenAI\'s GPT3 produces the best results.\n'
                         'EleutherAI\'s GPT-Neo models are a solid free '
                         'alternative.\nNaive is mostly for debugging and '
                         'will load a saved\nresponse from GPT3 for a simple '
                         'date formatting task.')

            with label_above('task_list', 'Tasks:'):
                add_listbox(
                    'task_list', items=list(NAME2TASK),
                    num_items=len(NAME2TASK), callback=task_select_callback,
                    callback_data={'task_list_id': 'task_list',
                                   'text_source_id': 'transcribed_text'}
                )

            with tooltip('task_list', 'task_list_tooltip'):
                add_text('Select a task to perform.\nAll of these accept some '
                         'input\ntext and produce new text,\noften in '
                         'response to or\nbased on the input. Tasks\nprefixed '
                         'by "(debug)" are\nunaffected by input.')

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
                add_input_text('stop', default_value='',
                               hint='GPT3 will stop producing text if it '
                                    'encounters any of these words/phrases, '
                                    'and they will be automatically excluded '
                                    'from the response.',
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
                               multiline=True,
                               width=self.widths[.5] - 2*self.pad, height=450)
            with tooltip('prompt', 'prompt_tooltip'):
                add_text('This is the full prompt that will\nbe sent to '
                         'GPT3. If you haven\'t\nrecorded or typed anything '
                         'in\nthe input box yet, you may\nsee a pair of curly '
                         'braces in here:\nthis will be replaced by\nyour '
                         'input once you provide it.')

        #######################################################################
        # Conversation Options Window
        #######################################################################
        with window('Conversation Options', width=self.widths[.5],
                    height=self.heights[1.],
                    x_pos=self.widths[.5] + 2*self.pad,
                    y_pos=self.pad, no_resize=True, no_move=True, show=False):

            # Make query.
            add_button(
                'conv_query_btn', label='Query',
                callback=conv_query_callback,
                callback_data={'target_id': 'conv_text',
                               'read_checkbox_id': 'conv_read_response',
                               'interrupt_id': 'conv_interrupt_checkbox',
                               'query_msg_id': 'conv_query_progress_msg'})
            add_same_line()
            add_checkbox('conv_read_response', label='Read Response',
                         default_value=True)
            add_same_line()
            add_input_int('speed_input', default_value=0, min_value=0,
                          max_value=10, min_clamped=True, max_clamped=True,
                          label='Speaker Speed',
                          width=int(APP.widths[.5] * .35),
                          callback=speaker_speed_callback)
            add_spacing(count=10)

            # Select a persona.
            add_button('add_persona_btn', label='Add Auto Persona',
                       callback=add_persona_callback,
                       callback_data={'name_id': 'add_persona_text',
                                      'target_id': 'persona_list',
                                      'show_during_id': 'add_persona_msg',
                                      'error_msg_id': 'add_persona_error_msg'})
            with tooltip('add_persona_btn', 'add_persona_btn_tooltip'):
                add_text('Make sure to finish your current conversation first '
                         '- \notherwise, this will end it automatically and '
                         'start a \nconversation with the new persona.')
            add_same_line()

            # name_source_id is only used by the first Add Custom Persona
            # button callback. This is just a default value - the user can
            # always enter one later in the modal that pops up.
            generate_persona_data = {
                'popup_id': 'Custom Persona Info',
                'name_id': 'persona_name',
                'summary_id': 'custom_summary',
                'image_path_id': 'custom_image_path',
                'gender_id': 'Gender',
                'target_id': 'persona_list',
                'error_msg_id': 'persona_save_error_msg',
                'name_source_id': 'add_persona_text',
                'force_save_id': 'force_generate_box'
            }
            add_button('add_custom_persona_btn', label='Add Custom Persona',
                       callback=add_custom_persona_callback,
                       callback_data=generate_persona_data)
            with tooltip('add_custom_persona_btn',
                         'add_custom_persona_btn_tooltip'):
                add_text('This lets you provide the persona\'s bio and '
                         'picture yourself \ninstead of auto-generating them.'
                         'Make sure to finish your\ncurrent conversation '
                         'first - otherwise, this will end it \nautomatically '
                         'and start a conversation with the new persona.')
            with popup('add_custom_persona_btn', 'Custom Persona Info',
                       modal=True, width=self.widths[.5],
                       mousebutton=mvMouseButton_Left):
                popup_item_width = int(self.widths[.5] * .85)
                add_input_text('persona_name', label='Name',
                               width=popup_item_width)
                add_radio_button('Gender', items=['F', 'M'],
                                 default_value=0,
                                 horizontal=True)
                add_input_text('custom_summary', label='Summary',
                               default_value='', multiline=True,
                               width=popup_item_width)
                add_input_text('custom_image_path', default_value='',
                               label='Image Path (optional)',
                               width=popup_item_width)
                add_button('custom_generate_btn', label='Generate',
                           callback=generate_persona_callback,
                           callback_data=generate_persona_data)
                add_same_line()
                add_button('custom_cancel_btn', label='Cancel',
                           callback=cancel_save_conversation_callback,
                           callback_data=generate_persona_data)
                add_same_line()
                add_checkbox('force_generate_box', label='Force Generate',
                             default_value=False)
                add_text('persona_save_error_msg',
                         default_value='Persona already exists.', show=False)

            add_same_line()
            add_text('add_persona_msg',
                     default_value='Generating new persona...', show=False)
            add_same_line()
            add_text('add_persona_error_msg',
                     default_value='Failed to generate that persona. Try a '
                                   'different name.', show=False)
            add_spacing(count=2)
            add_input_text('add_persona_text', label='')
            with label_above('persona_list', 'Existing Personas'):
                # Num_items ensures height stays constant even if we add more
                # personas. This does not need to be the current number of
                # available personas.
                add_listbox('persona_list',
                            items=CONV_MANAGER.personas(),
                            num_items=5,
                            callback=persona_select_callback)

            # Section info related to the active persona. We use a dummy
            # element to effectively center the image, which doesn't seem to be
            # natively supported (?).
            add_spacing(count=2)
            add_dummy(width=self.widths[.5] // 4, height=200,
                      name='img_dummy_spacer')
            # Don't worry about image or text kwargs much here - these are set
            # in update_persona_info(). Oddly, adding spacing between image and
            # text seems to break my dummy-powered centering method. Not a big
            # deal, but don't try to add that here unless you plan on a more
            # involved debugging session.
            add_image('conversation_img', str(CONV_MANAGER.current_img_path))
            add_text('summary_text', default_value='')
            update_persona_info()

    def build(self):
        """Create dearpygui windows and the elements they contain."""
        self.primary_window()
        self.left_column()
        self.right_column()

    def run(self):
        """Builds and runs dearpygui app."""
        self.build()
        start_dearpygui(primary_window='primary_window')


if __name__ == '__main__':
    # tasks = ['Barack Obama'] # TODO: tmp limit conv personas for faster loading.
    tasks = []
    SPEAKER = Speaker(newline_pause=400)
    CHUNKER = GuiTextChunker(max_chars=70)
    MANAGER = PromptManager(verbose=False, skip_tasks=['conversation',
                                                       'shortest',
                                                       'short_dates'])
    CONV_MANAGER = ConversationManager(tasks)
    NAME2TASK = IndexedDict({
        'Punctuate': 'punctuate',
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
    })
    MODEL_NAMES = ['gpt3', 'gpt-neo 2.7B', 'gpt-neo 1.3B', 'gpt-neo 125M',
                   'naive']
    GENDER2VOICE = {'F': 'karen',
                    'M': 'daniel'}

    # Create app, initialize callbacks, and run.
    APP = App()
    for module in ('callbacks', 'utils'):
        set_module_globals(module, SPEAKER=SPEAKER, CHUNKER=CHUNKER,
                           CONV_MANAGER=CONV_MANAGER, MANAGER=MANAGER,
                           NAME2TASK=NAME2TASK, MODEL_NAMES=MODEL_NAMES,
                           GENDER2VOICE=GENDER2VOICE, APP=APP)
    APP.run()
