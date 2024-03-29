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
from queue import Queue
import speech_recognition as sr
import time
from threading import Thread

from htools.core import save, select
from htools.meta import min_wait
from jabberwocky.openai_utils import GPTBackend

from utils import read_response_coro, stream, monitor_interrupt_checkbox, \
    CoroutinableThread, PropagatingThread, interrupt, img_dims


RECOGNIZER = sr.Recognizer()
RECOGNIZER.is_listening = False
RECOGNIZER.pause_threshold = 0.9
RECOGNIZER.calibrated = False


def transcribe(data, error_message, results):
    # User can cancel listener at any point within this try block.
    with sr.Microphone() as source:
        if not RECOGNIZER.calibrated:
            show_item(data['adjust_id'])
            RECOGNIZER.adjust_for_ambient_noise(source)
            RECOGNIZER.calibrated = True
            hide_item(data['adjust_id'])
        show_item(data['listening_id'])
        audio = RECOGNIZER.listen(source)
    try:
        text = RECOGNIZER.recognize_google(audio)
    except sr.UnknownValueError:
        text = error_message

    # Don't just use capitalize because this removes existing capitals.
    # Probably don't have these anyway (transcription seems to usually be
    # lowercase) but just being safe here.
    text = text[0].upper() + text[1:]
    results.append(text)
    return text


def transcribe_callback(sender, data):
    """Triggered when user hits the Record button. Used in both task mode
    and conv mode.

    data keys:
        - target_id (str: dearpygui text input box to display transcribed text
            in)
        - listening_id (str: dearpygui text message to display during
            transcription, usually something like "listening...".)
        - auto_punct_id (str: dearpygui checkbox determining whether to use
            gpt3 to auto-punctuate the transcribed text)
        - stop_record_id (str: dearpygui checkbox user can check to interrupt
            listening/transcription)
        - adjust_id (str: dearpygui text item displaying message while
            recognizer is adjusting for ambient noise. Currently set this up to
            execute only once per GUI session.)
        - pre_delete_dummy (bool: optional. If True (the default), try to
            delete dummy window if one exists before executing the
            transcription logic. This is created by our
            hotkey_handler to take focus away from an input text box so that we
            can update its output, which isn't possible if it's focused. This
            is False when triggered by the transcribe hotkey and True
            otherwise.)
        - post_delete dummy (bool: optional. Same as pre_delete_dummy but
            checks at the end of this function. At the moment this is always
            True but only executes when triggerd by our record callback, as
            that's the only time the dummy window should be present by the end
            of the function.)
    """
    if data.get('pre_delete_dummy', True) and does_item_exist('dummy window'):
        delete_item('dummy window')
        end()

    if is_item_visible('Input'):
        set_value(data['target_id'], '')
    error_message = 'Parsing failed. Please try again.'

    # Record until pause. Default is to stop recording when the speaker pauses
    # for 0.8 seconds, which I found a tiny bit short for my liking.
    RECOGNIZER.is_listening = True
    results = []
    thread_record = Thread(target=transcribe,
                           args=(data, error_message, results))
    thread_record.start()

    thread = PropagatingThread(target=monitor_interrupt_checkbox,
                               kwargs={'box_id': data['stop_record_id'],
                                       'errors': None,
                                       'obj': RECOGNIZER,
                                       'attr': 'is_listening',
                                       'wait': .25},
                               raise_immediately=True)
    thread.start()
    show_item(data['stop_record_id'])

    # Thread monitor will exit if the user checks the interrupt box or if
    # listening completes.
    while True:
        if not thread.is_alive():
            print('MONITOR THREAD DEAD. TERMINATING LISTENER.')
            interrupt(thread_record)
            break
        if not thread_record.is_alive():
            print('Record exited naturally.')
            break
        time.sleep(.1)
    thread_record.join()
    if results:
        text = results[0]
    else:
        text = ''

    # Separate this from the interruptable thread because it writes to a log
    # file and I haven't found a way to cleanup on exit. Only allow
    #  cancellation before or after this step. We do check if thread is alive
    # last to give the user up until the last possible fraction of a second to
    # cancel.
    if text and text != error_message and get_value(data['auto_punct_id']) \
            and thread.is_alive():
        log_debug('BEFORE transcribe: ' + text)
        _, text = MANAGER.query(task='punctuate_transcription', text=text,
                                stream=False, strip_output=True)
        log_debug('AFTER transcribe: ' + text)

    if not thread.is_alive():
        text = ''

    # Technically recognizer stopped listening earlier but we're just using
    # this to let our checkbox monitor know when to quit.
    RECOGNIZER.is_listening = False
    thread.join()

    # Cleanup various messages/widgets. Some may have been hidden earlier but
    # we make sure here in case the listening thread was interrupted early.
    set_value(data['stop_record_id'], False)
    hide_item(data['stop_record_id'])
    hide_item(data['listening_id'])
    hide_item(data['adjust_id'])

    # Update text and various components now that transcription is complete.
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
        CONV_MANAGER.query_later(text)
        # Do not use full_conversation here because we haven't added the new
        # user response yet.
        text = CONV_MANAGER._format_prompt(text, do_full=True,
                                           include_trailing_name=False,
                                           include_summary=False)
        chunked = CHUNKER.add('conv_transcribed', text)
        set_value(data['target_id'], chunked)

    if data.get('post_delete_dummy', True) and does_item_exist('dummy window'):
        delete_item('dummy window')
        end()


def format_text_callback(sender, data):
    """Used in task mode when user hits auto-format button. Mostly just
    chunks text since dearpygui doesn't wrap lines automatically. Also adds a
    colon at the end of text if task is "how to".

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
              'update_kwargs': data.get('update_kwargs', False)}
    )


@min_wait(1.5)
def hotkey_handler(sender, data):
    print('hotkey data:', data)
    if is_item_visible('Conversation'):
        conv_mode = True
    elif is_item_visible('Input'):
        conv_mode = False
    else:
        # Don't think this ever happens intentionally but I think it's
        # technically possible.
        print('Neither main window is visible.')
        return

    # Take focus off of the text input box where typing will occur in.
    # Otherwise hotkeys can't update text in that box if called when the cursor
    # is in that box, which is a common use case (e.g. in conv mode if we edit
    # our last transcription, we want to be able to call query without clicking
    # outside of the text input box. Hitting the escape key first doesn't work
    # either as it functions as an undo key in dearpygui.
    add_window('dummy window', no_focus_on_appearing=False,
               no_bring_to_front_on_focus=True, show=True)

    # CTRL + SHIFT: start recording if not already.
    if data == 340 and not RECOGNIZER.is_listening:
        if conv_mode:
            cb_data = {'listening_id': 'conv_record_msg',
                       'target_id': 'conv_text',
                       'auto_punct_id': 'conv_auto_punct',
                       'stop_record_id': 'conv_stop_record',
                       'adjust_id': 'conv_adjust_msg',
                       'pre_delete_dummy': False}
        else:
            cb_data = {'listening_id': 'record_msg',
                       'target_id': 'transcribed_text',
                       'auto_punct_id': 'auto_punct',
                       'stop_record_id': 'stop_record',
                       'adjust_id': 'adjust_msg',
                       'pre_delete_dummy': False}
        transcribe_callback('record_hotkey_callback', data=cb_data)

    # CTRL + a: query gpt3 for response.
    elif data == 65:
        if conv_mode:
            cb_data = {'target_id': 'conv_text',
                       'read_checkbox_id': 'conv_read_response',
                       'interrupt_id': 'conv_interrupt_checkbox',
                       'query_msg_id': 'conv_query_progress_msg',
                       'query_error_msg_id': 'conv_query_error_msg',
                       'engine_i_id': 'conv_engine_i_input'}
            conv_query_callback('record_hotkey_callback', cb_data)
        else:
            cb_data = {'target_id': 'response_text',
                       'read_checkbox_id': 'read_response',
                       'interrupt_id': 'interrupt_checkbox',
                       'query_msg_id': 'query_progress_msg'}
            query_callback('record_hotkey_callback', cb_data)

    # Transcribe callback does this, so this should already have happened
    # if this is triggered by a record hotkey or a query hotkey in conv mode.
    # If we use a query hotkey in task mode we need to cleanup though. I leave
    # this outside the if clauses in case the user hits CTRL with another
    # key.
    if does_item_exist('dummy window'):
        delete_item('dummy window')
        end()


def text_edit_callback(sender, data):
    """Triggered when user types in transcription text field. This way user
    edits update the prompt before making a query (this is often necessary
    since transcriptions are not always perfect).

    Note: set_key_press_callback() is undocumented but it can't pass in a dict
    as data (it's an int of unknown meaning).

    Here, sender is the id of the active window (e.g. Conversation, Options,
    etc.)
    """
    # 341 for ctrl
    hotkey = 341
    # User can hold CTRL and tap SHIFT to record. Data will be the most recent
    # key pressed, which we want to be shift (340). Therefore we check if
    # CTRL is also pressed.
    if is_key_down(hotkey):
        if data != hotkey: hotkey_handler('text_edit_callback', data)
        return
    # This is actually a different case than above: I'm guessing there are
    # times where this callback is triggered but by the time we reach the
    # is_key_down check CTRL has been released? Without this check we get
    # text edit errors in conv mode.
    if data == hotkey:
        return

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
    # triggered when user edits speech speed in task mode.
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
        pretty_name = CONV_MANAGER.process_name(
            CONV_MANAGER.current['persona'], inverse=True
        )
        if f'\n\n{pretty_name}: ' not in last_turn:
            CONV_MANAGER.query_later(last_turn)
        else:
            if not is_item_visible('edit_warning_msg'):
                show_item('edit_warning_msg')
                time.sleep(2)
                hide_item('edit_warning_msg')


def task_select_callback(sender, data):
    """Triggered when user selects a task (e.g. Summary) in task mode.

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
    name = CONV_MANAGER.current['persona']
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
    if data:
        name = data['name']
    else:
        name = CONV_MANAGER.personas()[get_value(sender)]

    # Avoid resetting vars in the middle of a conversation. Second part avoids
    # subtle issue where if we force generate a custom persona that's already
    # loaded (i.e. we overwrite an existing persona), the new persona wouldn't
    # auto-load. We don't need this for default add_persona action because
    # if the persona's already loaded, we'll just load it from files rather
    # than overwriting anything.
    if (CONV_MANAGER.process_name(name) == CONV_MANAGER.current['persona']) \
            and (sender != 'generate_persona_callback'):
        return

    CONV_MANAGER.start_conversation(name, download_if_necessary=False)
    update_persona_info()
    set_value('conv_text', '')
    # Must happen after we start conversation.
    SPEAKER.voice = GENDER2VOICE[CONV_MANAGER.current['gender']]

    if sender != 'end_conversation_callback':
        # Start listening for user response automatically.
        transcribe_callback('conv_query_callback',
                            {'listening_id': 'conv_record_msg',
                             'target_id': 'conv_text',
                             'auto_punct_id': 'conv_auto_punct',
                             'stop_record_id': 'conv_stop_record',
                             'adjust_id': 'conv_adjust_msg'})


def add_custom_persona_callback(sender, data):
    """Executes when user clicks Add Custom Persona button in conv mode. Note
    that the actual generation does not yet occur - that happens in
    generate_persona_callback. This just preps the form for the user to
    interact with.

    data keys:
      -  popup_id
      -  name_id
      -  summary_id
      -  image_path_id
      -  gender_id
      -  target_id
      -  error_msg_id
      -  name_source_id
      -  force_save_id
    """
    # Reset values so the next persona we generate doesn't show the last
    # persona we added.
    set_value(data['name_id'], get_value(data['name_source_id']))
    set_value(data['summary_id'], '')
    set_value(data['image_path_id'], '')
    set_value(data['gender_id'], 0)


def generate_persona_callback(sender, data):
    """Triggered when user clicks Generate button in popup window that appears
    after clicking Add Custom Persona from conv mode.
    """
    name = get_value(data['name_id']).strip(' ')
    summary = get_value(data['summary_id']).replace('\n', ' ')\
                                           .replace('  ', ' ')\
                                           .strip(' ')
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
    cancel_save_conversation_callback('generate_persona_callback', data)
    # Do this last since it triggers transcription callback.
    persona_select_callback('generate_persona_callback', {'name': name})


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
    name = get_value(data['name_id']).strip(' ')
    if not name:
        show_item(data['error_msg_id'])
        return
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
    finally:
        hide_item(data['show_during_id'])
        set_value(data['name_id'], '')

    # Make new persona the active one. Dearpygui doesn't seem to let us
    # change the selected listbox item so we have to do this a bit hackily.
    # Keep this outside of finally block because we don't want it to execute if
    # there was an error adding the persona.
    persona_select_callback('add_persona_callback', {'name': name})


def cancel_save_conversation_callback(sender, data):
    """This is executed when the user hits cancel in the Save file popup. It's
    also manually called at the end of save_callback and
    generate_persona_callback since it's a convenient way to reset several
    items at once.

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
    # Add Custom Persona has this. Covers case where we manually call this from
    # generate_persona_callback or when the user clicks Cancel button in the
    # popup that appears when adding a custom person.
    if 'name_source_id' in data:
        set_value(data['name_source_id'], '')
    close_popup(data['popup_id'])


def saveas_callback(sender, data):
    """Triggered when user hits Save As button in either task or conv mode.
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
        file = f'{CONV_MANAGER.current["persona"]}_{date}.txt'
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
        Task mode doesn't really need to specify them since they're unused
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
    # task mode in case we want to reuse a prompt for multiple tasks.
    if is_item_visible('Conversation') and get_value(data['end_conv_id']):
        if full_conv: CHUNKER.delete('conv_transcribed')
        set_value(data['source_text_id'], '')
        CONV_MANAGER.start_conversation(CONV_MANAGER.current['persona'])
    cancel_save_conversation_callback('save_callback', data)


def query_callback(sender, data):
    """Triggered when user hits query button in task mode. Conv query is
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
        kwargs = update_query_kwargs_from_model_name(model, kwargs)
        try:
            res = MANAGER.query(task=task, text=text, strip_output=False,
                                **kwargs)
            # In mode stream=False (i.e. when not using GPT3) we only want the
            # response, not the prompt.
            if isinstance(res, tuple): res = res[1]
        except Exception as e:
            print(e)
            res = 'Query failed. Please check your settings and try again.'

    # Type and read response aloud.
    concurrent_speaking_typing(res, data)


def update_query_kwargs_from_model_name(model, query_kwargs):
    """Adjust query parameters depending on which model a user chose. For
    example, GPT3 will use streaming mode, GPT neo query function requires a
    `size` param, etc.

    Parameters
    ----------
    model: str
        One of MODEL_NAMES defined in gui/main.py.
        E.g. GPT3, GPT-neo 1.3B, GPT-J.
    query_kwargs: dict
        Params controlling text generation. E.g. max_tokens, temperature.

    Returns
    -------
    dict: Same as query_kwargs but adjusted to use the appropriate model. Other
    kwargs are tweaked too since each API offers slightly different options.
    """
    model = model.lower()
    query_kwargs = dict(query_kwargs)
    if 'neo' in model:
        GPT.switch('huggingface')
        # ISSUE: update query_kwargs with new model name. Problem is new
        # jabberwocky only provides 2 huggingface engines, only 1 of which
        # overlaps with the 3 options in the GUI. (This made things more
        # consistent with the EngineMap interface - i.e. if we ask for model 3
        # we're probably expecting davinci-level models, which huggingface did
        # not provide at jabberwocky v2 development time.
    elif model == 'gpt-j':
        GPT.switch('banana')
    elif model == 'naive':
        GPT.switch('mock')
    else:
        # Only gpt3 mode really supports streaming. Naive technically does
        # but it's not particularly useful.
        query_kwargs['stream'] = True
    return query_kwargs


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

    # Query gpt3, then type and read response.
    show_item(data['query_msg_id'])
    model = MODEL_NAMES[get_value('conv_model')]
    kwargs = update_query_kwargs_from_model_name(
        model, {'engine_i': get_value(data['engine_i_id'])}
    )
    res = CONV_MANAGER.query(**kwargs)
    # In mode stream=False (i.e. when not using GPT3) we only want the
    # response, not the prompt.
    if isinstance(res, tuple): res = res[1]
    concurrent_speaking_typing(res, data, conv_mode=True)

    # Start listening for user response automatically.
    transcribe_callback('conv_query_callback',
                        {'listening_id': 'conv_record_msg',
                         'target_id': 'conv_text',
                         'auto_punct_id': 'conv_auto_punct',
                         'stop_record_id': 'conv_stop_record',
                         'adjust_id': 'conv_adjust_msg'})


def concurrent_speaking_typing(streamable, data, conv_mode=False, pause=.18):
    # Stream function provides "typing" effect.
    # full_text only used in task mode.
    full_text = ''
    errors = []
    q = Queue()
    monitor_thread = Thread(target=monitor_interrupt_checkbox,
                            kwargs={'box_id': data['interrupt_id'],
                                    'errors': errors,
                                    'obj': SPEAKER,
                                    'attr': 'is_speaking',
                                    'wait': .25})
    speaker_thread = CoroutinableThread(target=read_response_coro, queue=q,
                                        args=(data, errors))
    monitor_thread.start()
    speaker_thread.start()

    # Only used in conv mode - task mode must update full_text each
    # iteration regardless. ConvManager does it automatically when streaming
    # because we use the full_conversation attr, but when stream=False (as it
    # must be when using one of the open source models) we already have the
    # full conversation by the time this function is called. Therefore, without
    # this step we'd display the text all at once rather than getting our
    # typing effect.
    streamed_response = True
    if conv_mode and isinstance(streamable, str):
        full_text = CONV_MANAGER.full_conversation(include_summary=False)
        pretty_name = CONV_MANAGER.process_name(
            CONV_MANAGER.current['persona'], inverse=True
        )
        full_text = ''.join(full_text.rpartition(f'\n\n{pretty_name}: ')[:-1])
        streamed_response = False

    # Chunk should be a token when using GPT-3 and a word (space-split)
    # otherwise.
    for chunk in stream(streamable):
        if conv_mode:
            if streamed_response:
                full_text = CONV_MANAGER.full_conversation(False)
            else:
                full_text += chunk
            chunked = CHUNKER.add('conv_transcribed', full_text)
        else:
            full_text += chunk
            chunked = CHUNKER.add('response', full_text)
        set_value(data['target_id'], chunked)
        try:
            speaker_thread.queue.put(chunk)
        except StopIteration:
            pass

        time.sleep(pause)
    # Sentinel value ensures we speak any remaining sentence.
    try:
        speaker_thread.queue.put(None)
    except StopIteration:
        pass
    speaker_thread.join()
    monitor_thread.join()
    hide_item(data['interrupt_id'])
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
    set_item_label('task_menu_choice', 'Task')


def menu_task_callback(sender, data):
    """Triggered when user selects task mode from the main menu.
    Controls which windows are displayed.
    """
    show_item('Input')
    show_item('Options')
    hide_item('Conversation')
    hide_item('Conversation Options')
    set_item_label('task_menu_choice', 'Task\t[x]')
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
    dims = img_dims(CONV_MANAGER.current['img_path'],
                    width=(APP.widths[.5] - 2*APP.pad) // 2)
    set_item_width(dummy_name, APP.widths[.5] // 4 - 4*APP.pad)
    add_same_line(parent=parent, before=img_name)
    # Must delete image after previous updates.
    delete_item(img_name)
    add_image(img_name, str(CONV_MANAGER.current['img_path']), parent=parent,
              before=text_name, **dims)
    chunked = CHUNKER.add(text_key, CONV_MANAGER.current['summary'],
                          max_chars=dims['width'] // 4)
    set_value(text_name, chunked)


