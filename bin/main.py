# I think dearpygui imports a different contextmanager so we rename this one.
from contextlib import contextmanager as ctx_manager
from datetime import datetime as dt
from dearpygui.core import *
from dearpygui.simple import *
from nltk.tokenize import sent_tokenize
import os
from pathlib import Path
import speech_recognition as sr
import time
from threading import Thread

from htools.core import tolist, select, save, load
from htools.meta import params, debug, decorate_functions
from htools.structures import IndexedDict
from jabberwocky.openai_utils import PromptManager, ConversationManager,\
    query_gpt3, query_gpt_neo
from jabberwocky.speech import Speaker
from jabberwocky.core import GuiTextChunker
from jabberwocky.utils import img_dims


os.chdir('../')
SPEAKER = Speaker(newline_pause=400)
CHUNKER = GuiTextChunker(max_chars=70)
MANAGER = PromptManager(verbose=False,
                        skip_tasks=['conversation', 'shortest', 'short_dates'])
# TODO: eventually make all personas available but loading is faster this way
# which is nice during dev. Use > 1 to allow testing switching between
# personas.
CONV_MANAGER = ConversationManager('Barack Obama', 'Brandon Sanderson')

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
    'Dates (debug)': 'short_dates',
    'Math (debug)': 'shortest'
})
MODEL_NAMES = ['gpt3', 'gpt-neo 2.7B', 'gpt-neo 1.3B', 'gpt-neo 125M', 'naive']
GENDER2VOICE = {'F': 'karen',
                'M': 'daniel'}


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
        - is_default
    """
    if data['is_default']:
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

    # Update text and various components now that transcription is complete.
    for id_ in show_during:
        hide_item(id_)
    for id_ in data.get('show_after_ids', []):
        show_item(id_)

    if data['is_default']:
        set_value(data['target_id'], text)
        # If not in conversation mode, manually call this so prompt is updated
        # once we finish recording.
        format_text_callback('task_list',
                             data={'text_source_id': data['target_id'],
                                   'task_list_id': 'task_list',
                                   'update_kwargs': True,
                                   'key': 'transcribed'})
    else:
        CONV_MANAGER.query_later(text)
        text = CONV_MANAGER.format_prompt(text, exclude_trailing_name=True)
        chunked = CHUNKER.add('conv_transcribed', text)
        set_value(data['target_id'], chunked)


def format_text_callback(sender, data):
    """
    data keys:
        - text_source_id
        - key (str: name CHUNKER will use to map raw text to chunked text.)
        - task_list_id
        - update_kwargs (bool)
    """
    task_name = NAME2TASK[get_value(data['task_list_id'])]
    text = get_value(data['text_source_id'])

    # Auto-transcription doesn't add colon at end so we do it manually.
    if task_name == 'how_to' and text and not text.endswith(':'):
        text += ':'
    chunked = CHUNKER.add(data['key'], text)
    set_value(data['text_source_id'], chunked)

    task_select_callback(
        'task_list',
        data={'task_list_id': data['task_list_id'],
              'text_source_id': data['text_source_id'],
              'update_kwargs': data.get('update_kwargs', False)})


def text_edit_callback(sender, data):
    """Triggered when user types in transcription text field. This way user
    edits update the prompt before making a query (this is often necessary
    since transcriptions are not always perfect).

    Note: set_key_press_callback() is undocumented but it can't pass in a dict
    as data (it's an int of unknown meaning).
    """
    # This way even if user doesn't hit Auto-Format, query_callback() can
    # retrieve input from chunker. Otherwise we'd have to keep track of when to
    # retrieve it from text input box vs. from chunker which could get messy.
    if is_item_visible('default_window'):
        CHUNKER.add('transcribed', get_value('transcribed_text'),
                    return_chunked=False)
        task_select_callback('task_list',
                             data={'task_list_id': 'task_list',
                                   'text_source_id': 'transcribed_text',
                                   'update_kwargs': False})
    # Case: in conversation mode and user edits the add_persona text.
    elif sender == 'conv_options_window':
        # Can't pass data dict to this type of callback (seems to be a
        # character code for the last typed character instead) so we have to
        # hardcode this. This ensures that if we previously failed to add a
        # persona and started typing a new name in that box, the error message
        # will go away.
        hide_item('add_persona_error_msg')
    # Case: in conversation mode and user edits the main conversation text box.
    # Not sure if this is really effective though because the conv manager
    # maintains its own conversation history and I think it only uses CHUNKER
    # to get the new prompt.
    else:
        edited = get_value('conv_text')
        CHUNKER.add('conv_transcribed', edited, return_chunked=False)
        # Can't figure out a way to get around this: have to do some surgery
        # to extract the last user turn. You can't edit gpt3 speech or your
        # turns that have already been sent to the model. Editing your most
        # recently transcribed turn BEFORE querying is the only accepted edit
        # type.
        last_turn = edited.rpartition('\n\nMe: ')[-1].strip()
        pretty_name = CONV_MANAGER.process_name(CONV_MANAGER.current_persona,
                                                inverse=True)
        if f'\n\n{pretty_name}: ' not in last_turn:
            CONV_MANAGER.query_later(last_turn)
        else:
            show_item('edit_warning_msg')
            time.sleep(2)
            hide_item('edit_warning_msg')


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
    # TODO: Is this necessary? Can't remember and it looks repetitive, but I
    # vaguely recall adding this to fix some bug.
    if CHUNKER._previously_added('transcribed', user_text):
        user_text = CHUNKER.get('transcribed', chunked=False)
    updated_prompt = MANAGER.prompt(task_name, user_text)
    chunked_prompt = CHUNKER.add('prompt', updated_prompt)
    set_value('prompt', chunked_prompt)
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


def end_conversation_callback(sender, data):
    name = CONV_MANAGER.current_persona
    CONV_MANAGER.end_conversation()
    try:
        CHUNKER.delete('conv_transcribed')
    except KeyError:
        pass
    persona_select_callback('end_conversation_callback', data={'name': name})


def persona_select_callback(sender, data):
    """Basically the same as update_persona_info() except this also starts a
    conversation in the ConversationManager. That step doesn't need to be done
    every time we resize the page, but the other steps in update_persona_info()
    do.

    Sender is normally the persona listbox, but when adding a new persona we
    want to load it automatically but can't change the selected listbox item.
    We allow for this by passing in a `data` dict with the key "name" (the
    pretty-formatted name to load). Otherwise, I believe this arg is None or
    an empty dict - forget which but not important right now). The
    end_conversation_callback also uses this strategy to reset the
    conversation and accompanying metadata.
    """
    # Don't love hard-coding this but there's no data arg when triggered by
    # listbox selection.
    hide_item('add_persona_error_msg')
    if data:
        name = data['name']
    else:
        name = CONV_MANAGER.personas()[get_value(sender)]
    # Avoid resetting vars in the middle of a conversation.
    if CONV_MANAGER.process_name(name) == CONV_MANAGER.current_persona:
        return

    CONV_MANAGER.start_conversation(name, download_if_necessary=False)
    update_persona_info()
    set_value('conv_text', '')
    # Must happen after we start conversation.
    SPEAKER.voice = GENDER2VOICE[CONV_MANAGER.current_gender]


def update_persona_info(img_name='conversation_img',
                        parent='conv_options_window', text_name='summary_text',
                        text_key='summary', dummy_name='img_dummy_spacer'):
    """Update and resize persona bio and summary (characters per line changes
    for the latter). This operates in the conv_options_window.

    Parameters
    ----------
    img_name
    parent
    text_name
    text_key
    dummy_name

    Returns
    -------

    """
    dims = img_dims(CONV_MANAGER.current_img_path,
                    width=(app.widths[.5] - 2*app.pad) // 2)
    set_item_width(dummy_name, app.widths[.5] // 4 - 4*app.pad)
    add_same_line(parent=parent, before=img_name)
    # Must delete image after previous updates.
    # if does_item_exist(img_name):
    delete_item(img_name)
    add_image(img_name, str(CONV_MANAGER.current_img_path), parent=parent,
              before=text_name, **dims)
    chunked = CHUNKER.add(text_key, CONV_MANAGER.current_summary,
                          max_chars=dims['width'] // 4)
    set_value(text_name, chunked)


def add_persona_callback(sender, data):
    """
    data keys:
        source_id (str: name of text input where user enters a new name)
        target_id (str: name of listbox to update after downloading new data)
    """
    name = get_value(data['source_id'])
    if not name: return
    show_item(data['show_during_id'])
    try:
        CONV_MANAGER.add_persona(name)
    except RuntimeError as e:
        # This will be hidden again when the user selects another persona or
        # types in the add_persona text input.
        show_item(data['error_msg_id'])
        return
    else:
        personas = CONV_MANAGER.personas()
        configure_item(data['target_id'], items=personas)
        # Make new persona the active one. Dearpygui doesn't seem to let us
        # change the selected listbox item so we have to do this a bit hackily.
        persona_select_callback('add_persona_callback', {'name': name})
    finally:
        hide_item(data['show_during_id'])


def cancel_save_conversation_callback(sender, data):
    # Reset checkbox and error message to defaults before closing modal. Don't
    # need to reset dir name and file name here because that happens in Save As
    # callback.
    hide_item(data['error_msg_id'])
    set_value(data['force_save_id'], False)
    close_popup(data['popup_id'])


def saveas_callback(sender, data):
    """data keys:
        - dir_id (str: identifies dearpygui text input field for dir name)
        - file_id (str: identifies dearpygui text input field for file name)
        - task_list_id (str: DEFAULT MODE ONLY. Identifies dearpygui listbox
            item so we can retrieve the current task.)
    """
    # This executes when the user first clicks Save As.
    # save_callback will then execute if the user proceeds to hit
    # save from the newly visible modal.
    date = dt.today().strftime('%Y-%m-%d')
    if is_item_visible('conv_window'):
        dir_ = str(CONV_MANAGER.conversation_dir.absolute())
        file = f'{CONV_MANAGER.current_persona}_{date}.txt'
    else:
        task_name = NAME2TASK[get_value(data['task_list_id'])]

        dir_ = str(Path(f'data/completions/{task_name}').absolute())
        file = f'{date}.txt'
    set_value(data['dir_id'], dir_)
    set_value(data['file_id'], file)


def save_callback(sender, data):
    """
    source_text_id (str: The text fields containing the content being saved. If
        in conversation mode, these will be cleared after saving. Default mode
        doesn't really need to specify them since they're unused in that case.
    end_conv_id (str: ID of checkbox in conv mode specifying whether to end
        the conversation. This isn't always true - we might want to save as we
        go along, just like any time you're writing a long document. Only
        necessary in conv mode.)
    popup_id (str: ID of the popup opened by the saveas button. This will be
        passed to cancel_save_conversation_callback to close the modal.)
    error_msg_id: (str: ID of error message to display if we need to warn user
        that the file name already exists or that there's nothing to save)
    dir_id: (str: ID of text input box where user types directory
        name)
    file_id: (str: ID of text input box where user types file name)
    force_save_id (str: ID of dearpygui checkbox to force save despite any
        warnings that may have been surfaced)
    """
    # Don't use ConversationManager's built-in save functionality because the
    # running prompt is only updated with the user's last response when a query
    # is made. If the user is the last one to comment, that line would be
    # excluded from the saved conversation.
    path = os.path.join(get_value(data['dir_id']), get_value(data['file_id']))
    if os.path.exists(path) and not get_value(data['force_save_id']):
        set_value(data['error_msg_id'], 'File already exists.')
        show_item(data['error_msg_id'])
        return

    # In case user tries to save an empty conversation.
    try:
        if is_item_visible('conv_window'):
            full_conv = CHUNKER.get('conv_transcribed', chunked=False)
        else:
            # Some surgery required: don't want full prompt with all examples
            # but do want to include any relevant context included in the
            # prompt. For example, the mma prediction prompt includes the word
            # "Analysis:" at the end.
            task_name = NAME2TASK[get_value(data['task_list_id'])]
            prompt_template = MANAGER.prompts[task_name]['prompt']
            # Seems like rpartition provides 3 items even if sep isn't found,
            # while rsplit would only return 1 item in that case.
            examples, _, partial_response = prompt_template.rpartition('{}')
            partial_input = examples.split('\n\n')[-1]
            full_conv = (partial_input
                         + CHUNKER.get('transcribed', chunked=False)
                         + partial_response
                         + ' '
                         + CHUNKER.get('response', chunked=False))
    except KeyError:
        full_conv = ''
    if not full_conv and not get_value(data['force_save_id']):
        set_value(data['error_msg_id'], 'There is no conversation yet.')
        show_item(data['error_msg_id'])
        return
    save(full_conv + '\n', path)

    # Reset text box, text chunker, and conversation manager. These should NOT
    # be done if we cancel the save operation. If user saved empty text, delete
    # call would throw an error without if clause. We also don't do this for
    # default mode in case we want to reuse a prompt for multiple tasks.
    if is_item_visible('conv_window') and get_value(data['end_conv_id']):
        if full_conv: CHUNKER.delete('conv_transcribed')
        set_value(data['source_text_id'], '')
        CONV_MANAGER.start_conversation(CONV_MANAGER.current_persona)
    cancel_save_conversation_callback('save_callback', data)


def query_callback(sender, data):
    """data keys:
        - target_id (str: element to display text response in)
        - interrupt_id (str: button to interrupt speaker if enabled)
    """
    show_item(data['query_msg_id'])
    # Can't pass empty list in for stop parameter.
    kwargs = app.get_query_kwargs()
    kwargs['stop'] = kwargs['stop'] or None
    # In this case we don't want the resolved prompt. These kwargs will be
    # passed to our query manager and the version returned by get_query_kwargs
    # uses the chunked text.
    del kwargs['prompt']

    # Want to send gpt3 the version of text without extra newlines inserted.
    task, text = app.get_prompt_text(do_format=False)
    if not text:
        res = 'Please record or type something in the text input box before ' \
              'making a query.'
    else:
        text = CHUNKER.get('transcribed', chunked=False)

        model = MODEL_NAMES[get_value('model')]
        if 'neo' in model:
            kwargs.update(mock_func=query_gpt_neo, size=model.split()[-1],
                          mock=True)
        elif model == 'naive':
            kwargs.update(mock=True, mock_func=None)

        try:
            print('input prompt:\n' + text)
            _, res = MANAGER.query(task=task, text=text, **kwargs)
        except Exception as e:
            print(e)
            res = 'Query failed. Please check your settings and try again.'

    # GPT3 seems to like a type of apostrophe that dearpygui can't display. We
    # also insert newlines to display the text more nicely in the GUI, but
    # avoid overwriting the raw response because the newlines add pauses when
    # speech mode is enable.
    chunked = CHUNKER.add('response', res)
    set_value(data['target_id'], chunked)
    hide_item(data['query_msg_id'])
    print('\nres:\n' + chunked)
    if get_value(data['read_checkbox_id']):
        read_response(res, data)


def read_response(response, data):
    # Read response if desired. Threads allow us to interrupt speaker if user
    # checks a checkbox. This was surprisingly difficult - I settled on a
    # partial solution that can only quit after finishing saying a
    # sentence/line, so there may be a bit of a delayed response after asking
    # to interrupt.
    show_item(data['interrupt_id'])
    errors = []
    thread = Thread(target=monitor_interrupt_checkbox,
                    args=(data['interrupt_id'], errors))
    thread.start()
    for sent in sent_tokenize(response):
        for chunk in sent.split('\n\n'):
            SPEAKER.speak(chunk)
            if errors:
                set_value(data['interrupt_id'], False)
                break
    hide_item(data['interrupt_id'])
    thread.join()


def conv_query_callback(sender, data):
    """
    data keys:
        - target_id (str: text input element, used to display both input and
        output)
    """
    show_item(data['query_msg_id'])
    # Notice we track our chunked conversation with a single key here unlike
    # default mode, where we require 1 for transcribed inputs and 1 for
    # GPT3-generated outputs.
    # Grab response to use when reading lines, otherwise we have to perform
    # "surgery" on full conv.
    # TODO: rm engine=0 after testing
    _, response = CONV_MANAGER.query(engine_i=0)
    hide_item(data['query_msg_id'])
    full_conv = CHUNKER.add('conv_transcribed', CONV_MANAGER.full_conversation)
    set_value(data['target_id'], full_conv)
    if get_value(data['read_checkbox_id']):
        read_response(response, data)


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
    # Resize callback is one of the few to not accept data.
    width, height = get_main_window_size()
    app.recompute_dimensions(width, height)
    windows = ['conv_window', 'default_window', 'default_options_window',
               'conv_options_window']
    for i, id_ in zip([0, 0, 1, 1], windows):
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

    # Images don't show up as children so we resize them separately.
    if is_item_visible('conv_options_window'):
        update_persona_info()


def menu_conversation_callback(sender, data):
    show_item('conv_window')
    show_item('conv_options_window')
    hide_item('default_window')
    hide_item('default_options_window')
    set_item_label('conv_menu_choice', 'Conversation\t[x]')
    set_item_label('default_menu_choice', 'Default')


def menu_default_callback(sender, data):
    show_item('default_window')
    show_item('default_options_window')
    hide_item('conv_window')
    hide_item('conv_options_window')
    set_item_label('default_menu_choice', 'Default\t[x]')
    set_item_label('conv_menu_choice', 'Conversation')


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
        self.menu_height = 25
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
        # account for 2. Height computations are slightly different due to
        # global menu bar.
        # pct2size = {p: int((dim - ((3-i)*self.pad)) * p)
        #             for i, p in enumerate(pcts)}
        # if mode == 'height':
        #     pct2size = {k: v - self.menu_height for k, v in pct2size.items()}
        # return pct2size

        pct2size = {}
        is_height = mode == 'height'
        for i, p in enumerate(pcts):
            size = dim - ((3-i)*self.pad) - self.menu_height*is_height
            pct2size[p] = int(size * p)
        return pct2size

    def recompute_dimensions(self, width, height):
        """
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
        pos = [
            [(self.pad, self.pad + self.menu_height),
             (self.widths[.5] + 2*self.pad, self.pad + self.menu_height)],
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
        if do_format:
            return MANAGER.prompt(task_name, text=input_)
        return task_name, input_

    def get_query_kwargs(self):
        # Gets currently selected kwargs from GUI.
        kwargs = {name: get_value(name) for name in self.query_kwarg_ids}
        kwargs['stop'] = kwargs['stop'].splitlines()
        kwargs['prompt'] = self.get_prompt_text()
        return kwargs

    def primary_window(self):
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

    def left_column(self):
        with window('default_window', width=self.widths[.5],
                    height=self.heights[1.], x_pos=self.pad,
                    y_pos=self.pad, no_resize=True, no_move=True):
            ###################################################################
            # Default window (1 time queries)
            ###################################################################
            add_button('record_btn', label='Record',
                       callback_data={'show_during_ids': ['record_msg'],
                                      'target_id': 'transcribed_text',
                                      'is_default': True},
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
        with window('conv_window', width=self.widths[.5],
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
                                      'target_id': 'conv_text',
                                      'is_default': False},
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
            add_button('end_conversation_btn', label='End Conversation',
                       callback=end_conversation_callback)
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
        with window('default_options_window', width=self.widths[.5],
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
        with window('conv_options_window', width=self.widths[.5],
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
            add_checkbox('conv_read_response', label='read response',
                         default_value=True)

            # Select a persona.
            add_button('add_persona_btn', label='Add Persona',
                       callback=add_persona_callback,
                       callback_data={'source_id': 'add_persona_text',
                                      'target_id': 'persona_list',
                                      'show_during_id': 'add_persona_msg',
                                      'error_msg_id': 'add_persona_error_msg'})
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
        self.primary_window()
        self.left_column()
        self.right_column()

    def run(self):
        self.build()
        start_dearpygui(primary_window='primary_window')


if __name__ == '__main__':
    app = App()
    app.run()
