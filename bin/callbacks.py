"""Callbacks for our dearpygui app. Note that this module relies on a number of
global variables which are defined after import in main.py:
-APP
-SPEAKER
-CHUNKER
-CONV_MANAGER
-MANAGER
-GENDER2VOICE
-MODEL_NAMES
-NAME2TASK

Pycharm highlights these as errors since they appear to be undefined when
looking at this module in isolation.
"""

from datetime import datetime as dt
from dearpygui.core import *
from dearpygui.simple import *
import os
from pathlib import Path
import speech_recognition as sr
import time
from threading import Thread

from htools.core import save, select
from jabberwocky.openai_utils import query_gpt_neo
from jabberwocky.utils import img_dims

from utils import read_response, stream


def transcribe_callback(sender, data):
    """Triggered when user hits the Record button. Used in both default mode
    and conv mode.

    data keys:
        - target_id (str: dearpygui text input box to display transcribed text
            in)
        - show_during_ids (Iterable[str]: dearpygui items to display during
            transcription. Usually just a text element saying transcription is
            in progress.)
    """
    if is_item_visible('Input'):
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

    if is_item_visible('Input'):
        set_value(data['target_id'], text)
        # If not in conversation mode, manually call this so prompt is updated
        # once we finish recording.
        format_text_callback('task_list',
                             data={'text_source_id': data['target_id'],
                                   'task_list_id': 'task_list',
                                   'update_kwargs': True,
                                   'key': 'transcribed'})
    else:
        # CONV_MANAGER.query_later(text)
        # # Do not use full_conversation here because we haven't added the new
        # # user response yet.
        # text = CONV_MANAGER._format_prompt(text, do_full=True,
        #                                    include_trailing_name=False,
        #                                    include_summary=False)
        # chunked = CHUNKER.add('conv_transcribed', text)
        # set_value(data['target_id'], chunked)

        # TODO: testing auto punctuate. Working old code is above.
        # NOTE: using cheap punctuation engine for testing but this doesn't
        # work that well. Engine_i=3 empirically works well, 0-1 does not. Need
        # test out i=2.
        print('BEFORE:', text)
        _, text = MANAGER.query(task='punctuate', text=text, stream=False,
                                strip_output=True, engine_i=0)
        print('AFTER:', text)
        # res = MANAGER.query(task=task, text=text, stream=True,
        #                     strip_output=False, **kwargs)
        CONV_MANAGER.query_later(text)
        # Do not use full_conversation here because we haven't added the new
        # user response yet.
        text = CONV_MANAGER._format_prompt(text, do_full=True,
                                           include_trailing_name=False,
                                           include_summary=False)
        chunked = CHUNKER.add('conv_transcribed', text)
        set_value(data['target_id'], chunked)


def format_text_callback(sender, data):
    """Used in conv mode when user hits auto-format button. Mostly just chunks
    text since dearpygui doesn't wrap lines automatically. Also adds a colon
    at the end of text if task is "how to".

    data keys:
        - text_source_id (str: name of element to get text from)
        - key (str: name CHUNKER will use to map raw text to chunked text)
        - task_list_id (str: dearpygui listbox to get current task from)
        - update_kwargs (bool: Whether to update kwargs in
            task_select_callback. False by default but True when this is
            called by transcribe_callback)
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

    Here, sender is the id of the active window (e.g. Conversation, Options,
    etc.)
    """
    # This way even if user doesn't hit Auto-Format, query_callback() can
    # retrieve input from chunker. Otherwise we'd have to keep track of when to
    # retrieve it from text input box vs. from chunker which could get messy.
    if is_item_visible('Input') and sender != 'Options':
        CHUNKER.add('transcribed', get_value('transcribed_text'),
                    return_chunked=False)
        task_select_callback('task_list',
                             data={'task_list_id': 'task_list',
                                   'text_source_id': 'transcribed_text',
                                   'update_kwargs': False})
    # Case: in conversation mode and user edits the add_persona text, Engine,
    # or speaker speed.
    elif sender == 'Conversation Options':
        # Can't pass data dict to this type of callback (seems to be a
        # character code for the last typed character instead) so we have to
        # hardcode this. This ensures that if we previously failed to add a
        # persona and started typing a new name in that box, the error message
        # will go away.
        hide_item('add_persona_error_msg')
    # Case: in conversation mode and user edits the main conversation text box.
    # Not sure if this is really effective though because the conv manager
    # maintains its own conversation history and I think it only uses CHUNKER
    # to get the new prompt. Use elif (not else) to prevent this from being
    # triggered when user edits speech speed in default mode.
    elif is_item_visible('Conversation'):
        edited = get_value('conv_text')
        CHUNKER.add('conv_transcribed', edited, return_chunked=False)
        # Can't figure out a way to get around this: have to do some surgery
        # to extract the last user turn. You can't edit gpt3 speech or your
        # turns that have already been sent to the model. Editing your most
        # recently transcribed turn BEFORE querying is the only accepted edit
        # type. If the edited turn is the user's first, we prefix the
        # conversation with a double newline so our parsing logic holds: on
        # subsequent turns the double newline will be automatically inserted
        # but the first one only has it when the summary is displayed, which
        # we've chosen to skip. This meant we were getting a tricky bug where
        # the first user turn was getting converted incorrectly: for example,
        # if we changed the text from "Me: Hi" to "Me: Hi.", then the text
        # edit callback would change our cached query to "Me: Me: Hi." because
        # rpartition would not find any "\n\nMe: " and thus the whole
        # transcription ended up in the "keep" portion. We could also remove
        # the double newline and just partition on "Me: " or even "Me:" but
        # I worry this text parsing is already a little brittle so I'd rather
        # not risk making it moreso.
        if not CONV_MANAGER.gpt3_turns and not edited.startswith('\n\n'):
            edited = '\n\n' + edited
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
    """Triggered when user selects a task (e.g. Summary) in default mode.

    data keys:
        - task_list_id (str: element containing selected item. Returns an int.)
        - text_source_id (str: element containing text for prompt input)
        - update_kwargs (bool: specifies whether to update query kwargs like
         max_tokens. We want those to be updated when we change the task but
             not when we manually call this function from our
             text_edit_callback.)
    """
    task_name, user_text = APP.get_prompt_text(
        task_list_id=data['task_list_id'],
        text_source_id=data['text_source_id'],
        do_format=False
    )
    if CHUNKER._previously_added('transcribed', user_text):
        user_text = CHUNKER.get('transcribed', chunked=False)
    updated_prompt = MANAGER.prompt(task_name, user_text)
    chunked_prompt = CHUNKER.add('prompt', updated_prompt)
    set_value('prompt', chunked_prompt)

    # Can't just use APP.get_query_kwargs() because that merely retrieves what
    # the GUI currently shows. We want the default kwargs which are stored by
    # our prompt manager.
    kwargs = MANAGER.kwargs(task_name)
    kwargs.setdefault('stop', '')
    # Fixed value for max_tokens doesn't make sense for this task.
    if task_name == 'punctuate':
        kwargs['max_tokens'] = int(len(user_text.split()) * 2)

        # Previously returned before even getting kwargs in this case, but
        # realized this means punctuation task max_tokens kwarg doesn't get
        # updated as we type. With updated system, that 1 kwarg will always
        # be updated. `updated_kwargs` controls the other kwargs since we don't
        # want those to be reset every time we type.
        if not data.get('update_kwargs', True):
            kwargs = select(kwargs, ['max_tokens'])
    for k, v in kwargs.items():
        # Choice of whether to mock calls is more related to the purpose of the
        # user session than the currently selected prompt.
        if k == 'mock':
            continue
        if k == 'stop' and isinstance(v, list):
            v = '\n'.join(term.encode('unicode_escape').decode() for term in v)
        set_value(k, v)


def end_conversation_callback(sender, data):
    """Triggered when user clicks the end conversation button in conv mode.
    Also tries to delete a stored conversation from CHUNKER if one exists.
    """
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

    data keys:
        - name (str: OPTIONAL. Usually this is determined by the selected item
            in our persona listbox, but when loading a new persona we want to
            force change the current person. However, dearpygui seemingly won't
            let us change the selected listbox item so I pass in the name
            manually).
    """
    # Don't love hard-coding this but there's no data arg when triggered by
    # listbox selection.
    hide_item('add_persona_error_msg')
    if data: name = data['name']
    else:
        name = CONV_MANAGER.personas()[get_value(sender)]
    # Avoid resetting vars in the middle of a conversation. Second part avoids
    # subtle issue where if we force generate a custom persona that's already
    # loaded (i.e. we overwrite an existing persona), the new persona wouldn't
    # auto-load. We don't need this for default add_persona action because
    # if the persona's already loaded, we'll just load it from files rather
    # than overwriting anything.
    if (CONV_MANAGER.process_name(name) == CONV_MANAGER.current_persona) and \
            (sender != 'generate_persona_callback'):
        return

    CONV_MANAGER.start_conversation(name, download_if_necessary=False)
    update_persona_info()
    set_value('conv_text', '')
    # Must happen after we start conversation.
    SPEAKER.voice = GENDER2VOICE[CONV_MANAGER.current_gender]


def add_custom_persona_callback(sender, data):
    # Reset values so the next persona we generate doesn't show the last
    # persona we added.
    set_value(data['name_id'], get_value(data['name_source_id']))
    set_value(data['summary_id'], '')
    set_value(data['image_path_id'], '')
    set_value(data['gender_id'], 0)


def generate_persona_callback(sender, data):
    name = get_value(data['name_id'])
    summary = get_value(data['summary_id']).replace('\n', ' ')\
                                           .replace('  ', ' ')
    img_path = get_value(data['image_path_id'])
    gender = ['F', 'M'][get_value(data['gender_id'])]
    already_exists = CONV_MANAGER.persona_exists_locally(name)
    force_save = get_value(data['force_save_id'])
    show_error = False
    if not (name and summary):
        set_value(data['error_msg_id'],
                  'Name and summary most both be provided.')
        show_error = True
    elif name in CONV_MANAGER and not force_save:
        set_value(data['error_msg_id'],
                  'Persona already loaded. Are you sure you want to overwrite '
                  'its summary and/or image path?')
        show_error = True
    elif already_exists and (summary or img_path) and not force_save:
        set_value(data['error_msg_id'],
                  'Persona already exists. Are you sure you want to '
                  'overwrite its summary and/or image path?')
        show_error = True
    if show_error:
        show_item(data['error_msg_id'])
        return

    try:
        CONV_MANAGER.add_persona(name, summary=summary, img_path=img_path,
                                 gender=gender, is_custom=True)
    except FileNotFoundError as e:
        set_value(data['error_msg_id'], 'Image path not found.')
        show_item(data['error_msg_id'])
        return

    # Update available personas in GUI and then make the new persona the active
    # one. Dearpygui doesn't seem to let us change the selected listbox item
    # so we have to do this a bit hackily.
    configure_item(data['target_id'], items=CONV_MANAGER.personas())
    persona_select_callback('generate_persona_callback', {'name': name})
    cancel_save_conversation_callback('generate_persona_callback', data)


def add_persona_callback(sender, data):
    """Triggered in conv mode when user clicks the Add Persona button. This
    usually requires internet access since we try to download a quick bio from
    Wikipedia if we haven't already.

    data keys:
        name_id (str: name of text input where user enters a new name)
        target_id (str: name of listbox to update after downloading new data)
        show_during_id (str: item to show while this executes. Usually a
            message explaining that downloading is occurring since it can take
            a few seconds)
    """
    name = get_value(data['name_id'])
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
    """This is executed when the user hits cancel in the Save file popup. It's
    also manually called at the end of save_callback since it's a convenient
    way to reset several items at once.

    data keys:
        - error_msg_id (str: text element containing a message displayed when
            a save error occurs. This will be hidden since we don't want it to
            show by default on our next save attempt.)
        - force_save_id (bool: checkbox element asking user whether they want
            to force save despite possible error (e.g. empty conversation or
            overwriting existing file name). This will also be reset.)
        - popup_id (str: the dearpygui popup item containing all save options.
            This is what finally closes the modal.)
    """
    # Resets checkbox and error message to defaults before closing modal.
    # Don't need to reset dir name and file name here because that happens in
    # Save As callback.
    hide_item(data['error_msg_id'])
    if 'force_save_id' in data:
        set_value(data['force_save_id'], False)
    close_popup(data['popup_id'])


def saveas_callback(sender, data):
    """Triggered when user hits Save As button in either default or conv mode.
    Note that saving will not actually occur until the user hits the save
    button in the popup that appears because we still need to confirm the file
    name, directory, etc.

    data keys:
        - dir_id (str: identifies dearpygui text input field for dir name)
        - file_id (str: identifies dearpygui text input field for file name)
        - task_list_id (str: DEFAULT MODE ONLY. Identifies dearpygui listbox
            item so we can retrieve the current task.)
    """
    # This executes when the user first clicks Save As.
    # save_callback will then execute if the user proceeds to hit
    # save from the newly visible modal.
    date = dt.today().strftime('%Y-%m-%d')
    if is_item_visible('Conversation'):
        dir_ = str(CONV_MANAGER.conversation_dir.absolute())
        file = f'{CONV_MANAGER.current_persona}_{date}.txt'
    else:
        task_name = NAME2TASK[get_value(data['task_list_id'])]
        dir_ = str(Path(f'data/completions/{task_name}').absolute())
        file = f'{date}.txt'
    set_value(data['dir_id'], dir_)
    set_value(data['file_id'], file)


def save_callback(sender, data):
    """Triggered when user hits save button inside the popup that appears after
    hitting the Save As button.

    data_keys:
        - source_text_id (str: The text fields containing the content being
        saved. If in conversation mode, these will be cleared after saving.
        Default mode doesn't really need to specify them since they're unused
        in that case. end_conv_id (str: ID of checkbox in conv mode specifying
        whether to end the conversation. This isn't always true - we might want
        to save as we go along, just like any time you're writing a long
        document. Only necessary in conv mode.)
        - popup_id (str: ID of the popup opened by the saveas button. This will
        be passed to cancel_save_conversation_callback to close the modal.)
        - error_msg_id: (str: ID of error message to display if we need to warn
        user that the file name already exists or that there's nothing to save)
        - dir_id: (str: ID of text input box where user types directory name)
        - file_id: (str: ID of text input box where user types file name)
        - force_save_id (str: ID of dearpygui checkbox to force save despite
        any warnings that may have been surfaced)
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
        if is_item_visible('Conversation'):
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
    if is_item_visible('Conversation') and get_value(data['end_conv_id']):
        if full_conv: CHUNKER.delete('conv_transcribed')
        set_value(data['source_text_id'], '')
        CONV_MANAGER.start_conversation(CONV_MANAGER.current_persona)
    cancel_save_conversation_callback('save_callback', data)


def query_callback(sender, data):
    """Triggered when user hits query button in default mode. Conv query is
    separate because the process is quite different: that's why we have a
    separate prompt manager for regular tasks and for conversation.

    data keys:
        - target_id (str: element to display text response in)
        - interrupt_id (str: button to interrupt speaker if enabled)
        - query_msg_id (str: name of text element to display during query since
            this takes a few seconds)
    """
    show_item(data['query_msg_id'])
    # Can't pass empty list in for stop parameter.
    kwargs = APP.get_query_kwargs()
    kwargs['stop'] = kwargs['stop'] or None
    # In this case we don't want the resolved prompt. These kwargs will be
    # passed to our query manager and the version returned by get_query_kwargs
    # uses the chunked text.
    del kwargs['prompt']

    # Want to send gpt3 the version of text without extra newlines inserted.
    task, text = APP.get_prompt_text(do_format=False)
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
            res = MANAGER.query(task=task, text=text, stream=True,
                                strip_output=False, **kwargs)
        except Exception as e:
            print(e)
            res = 'Query failed. Please check your settings and try again.'

    # Stream function provides "typing" effect.
    threads = []
    errors = []
    res_text = ''
    curr_text = ''
    for chunk in stream(res):
        res_text += chunk
        curr_text += chunk
        chunked = CHUNKER.add('response', res_text)
        set_value(data['target_id'], chunked)
        if any(char in chunk for char in ('.', '!', '?', '\n\n')):
            if not errors:
                thread = Thread(target=read_response,
                                args=(curr_text, data, errors, False))
                thread.start()
                threads.append(thread)
            # Make sure this isn't reset until AFTER the speaker thread starts.
            curr_text = ''
        time.sleep(.18)
    if curr_text and not errors:
        read_response(curr_text, data)
    hide_item(data['interrupt_id'])
    hide_item(data['query_msg_id'])
    for thread in threads: thread.join()


def conv_query_callback(sender, data):
    """Triggered when user hits query button in conv mode.

    data keys:
        - target_id (str: text input element, used to display both input and
        output)
        - read_checkbox_id (str: name of checkbox where user selects whether
            to read response aloud)
        - interrupt_id (str: name of checkbox user can check to interrupt
            Speaker mid-speech)
        - query_msg_id (str: name of text item containing the message to
            display while a query is in progress)
        - engine_i_id (str: name of int input where user selects engine_i)
        - no_query_msg_id (str: name of text item showing error message if user
            attempts to make a query before saying anything)
    """
    if not CONV_MANAGER.cached_query:
        show_item(data['query_error_msg_id'])
        time.sleep(1)
        hide_item(data['query_error_msg_id'])
        return

    show_item(data['query_msg_id'])
    # Notice we track our chunked conversation with a single key here unlike
    # default mode, where we require 1 for transcribed inputs and 1 for
    # GPT3-generated outputs.
    response = ''
    for chunk in CONV_MANAGER.query(engine_i=get_value(data['engine_i_id']),
                                    stream=True):
        full_conv = CHUNKER.add(
            'conv_transcribed',
            CONV_MANAGER.full_conversation(include_summary=False)
        )
        set_value(data['target_id'], full_conv)
        response += chunk

        # "Type" a bit faster than in default mode since we leave speaking til
        # the end. Most responses are only 1-2 sentences anyway so I feel its
        # not worth the added complexity to try to begin speaking sooner (we
        # need a full sentence or speech comes out sounding stilted).
        time.sleep(.16)
    if get_value(data['read_checkbox_id']):
        read_response(response, data)
    hide_item(data['query_msg_id'])


def speaker_speed_callback(sender, data):
    SPEAKER.rate = get_value(sender)


def resize_callback(sender):
    """Triggered when the GUI main window is resized. It's one of the few
    callbacks that doesn't accept a `data` argument.
    """
    width, height = get_main_window_size()
    APP.recompute_dimensions(width, height)
    windows = ['Conversation', 'Input', 'Options',
               'Conversation Options']
    for i, id_ in zip([0, 0, 1, 1], windows):
        set_item_width(id_, APP.widths[.5])
        set_item_height(id_, APP.heights[1.])
        set_window_pos(id_, *APP.pos[0][i])

        # 8 padding lengths seems to eliminate horizontal scrolling. Don't
        # fully understand why (doesn't quite match my calculations) but just
        # go with it.
        change_types = {'InputText', 'Listbox'}
        for child in get_item_children(id_):
            if get_item_type(child).split('::')[-1] in change_types:
                set_item_width(child, APP.widths[.5] - 8*APP.pad)

    # Images don't show up as children so we resize them separately.
    if is_item_visible('Conversation Options'):
        update_persona_info()


def menu_conversation_callback(sender, data):
    """Triggered when user selects Conversation mode from the main menu.
    Controls which windows are displayed.
    """
    show_item('Conversation')
    show_item('Conversation Options')
    hide_item('Input')
    hide_item('Options')
    set_item_label('conv_menu_choice', 'Conversation\t[x]')
    set_item_label('default_menu_choice', 'Default')


def menu_default_callback(sender, data):
    """Triggered when user selects default mode from the main menu.
    Controls which windows are displayed.
    """
    show_item('Input')
    show_item('Options')
    hide_item('Conversation')
    hide_item('Conversation Options')
    set_item_label('default_menu_choice', 'Default\t[x]')
    set_item_label('conv_menu_choice', 'Conversation')


def update_persona_info(img_name='conversation_img',
                        parent='Conversation Options',
                        text_name='summary_text',
                        text_key='summary',
                        dummy_name='img_dummy_spacer'):
    """Update and resize persona bio and summary (characters per line changes
    for the latter). This operates in the Conversation Options window.

    Parameters
    ----------
    img_name: str
        Name of dearpygui image element displaying current persona's photo.
    parent: str
        Parent dearpygui element (window) housing the summary.
    text_name: str
        Name of element displaying current persona summary.
    text_key: str
        The name to use when placing the summary in CHUNKER.
    dummy_name: str
        Name of dummy spacer element. It's size is set here.
    """
    dims = img_dims(CONV_MANAGER.current_img_path,
                    width=(APP.widths[.5] - 2*APP.pad) // 2)
    set_item_width(dummy_name, APP.widths[.5] // 4 - 4*APP.pad)
    add_same_line(parent=parent, before=img_name)
    # Must delete image after previous updates.
    delete_item(img_name)
    add_image(img_name, str(CONV_MANAGER.current_img_path), parent=parent,
              before=text_name, **dims)
    chunked = CHUNKER.add(text_key, CONV_MANAGER.current_summary,
                          max_chars=dims['width'] // 4)
    set_value(text_name, chunked)
