"""
WIP: trying Flask-Ask since I'm not sure how to run the lambda_function locally
of if that's even possible. App relies on local filesystem a lot at the moment
so it might be more convenient to run locally with ngrok anyway.
"""

import argparse
import ast
from datetime import datetime
import logging
from pathlib import Path
from flask import Flask, request
from flask_ask import question, context, statement
import numpy as np
# openai appears unused but is actually used by GPTBackend instance.
import openai
import requests
from transformers import pipeline

from config import EMAIL, LOG_FILE
from htools import quickmail, save, tolist, listlike, decorate_functions,\
    debug as debug_decorator, load
from jabberwocky.openai_utils import ConversationManager, PromptManager, GPT
from utils import slot, Settings, model_type, CustomAsk, infer_intent, voice, \
    custom_question, select_polly_voice


# Define these before functions since endpoints use ask method as decorators.
# This level unclutters terminal output a bit (it's hard to debug otherwise).
logging.getLogger('flask_ask').setLevel(logging.WARNING)
app = Flask(__name__)
# Necessary to make session accessible outside endpoint functions.
app.app_context().push()
state = Settings()
ask = CustomAsk(app=app, route='/', state=state, log_file=LOG_FILE, filler=' ')


def get_user_info(attrs=('name', 'email')):
    """Get user's email using Amazon-provided API. Obviously only works if
    they've approved access to this information.

    Returns
    -------
    str: User email if user gave permission and everything worked, empty string
    otherwise.
    """
    attrs = tolist(attrs)
    system = context.System
    token = system.get("apiAccessToken")
    endpoint = system.get('apiEndpoint')
    if not (token and endpoint):
        return ''
    res = dict.fromkeys(attrs, '')
    attr2suff = {
        'name': '/v2/accounts/~current/settings/Profile.givenName',
        'email': '/v2/accounts/~current/settings/Profile.email'
    }
    headers = {'Authorization': f'Bearer {token}'}
    for attr in attrs:
        r = requests.get(f'{endpoint}{attr2suff[attr]}', headers=headers)
        if r.status_code != 200:
            ask.logger.error(f'Failed to retrieve {attr}. Status code='
                             f'{r.status_code}, reason={r.reason}')
        else:
            res[attr] = r.json()
    return res


@app.route('/health')
def health():
    return {'status': 200}


# TODO: do I need to ask this somewhere? I enabled permission for MY account
# in alexa app and alexa site already but this is probably needed if I want to
# let someone else use it. Right now, the card is displayed after returning
# this (good) but if I respond Yes or No I just get an "audio response" (beep)
# from alexa. Unsure what this means.
@ask.intent('emailMe')
def tmp_email_me():
    return statement(
        'Do you mind if I access your name and email address? This will let '
        'me send you transcripts of your conversations. '
        'Say "I consent" or "no".'
    ).consent_card('alexa::profile:email:read')


def send_transcript(conv, user_email='', cleanup=False):
    """Email user a transcript of their conversation.

    Parameters
    ----------
    conv: ConversationManager
    user_email: str
        Email to send transcript to. If not provided, we retrieve it using
        an AWS api (prior user permission is required for this to work, of
        course). Could also retrieve this once, store it globally, and pass it
        in to prevent additional API calls and save a little time, but that's
        not a big priority since this isn't required every conversational turn.
    cleanup: bool
        If True, delete the file after sending the transcript. Otherwise, leave
        it in alexa/conversations.

    Returns
    -------
    bool: True if email was sent successfully, False otherwise (e.g. if no
    conversation has taken place yet or if the user has not provided us with
    their email).
    """
    user_email = user_email or get_user_info('email')['email']
    if not user_email:
        return False
    date = datetime.today().strftime('%m/%d/%Y')
    datetime_ = datetime.today().strftime('%Y.%m.%d__%H.%M.%S')
    tmp_path = Path(
        f'alexa/conversations/{conv.current["persona"]}__{datetime_}.txt'
    )
    save(conv.full_conversation(), tmp_path)
    name = conv.process_name(conv.current["persona"], inverse=True)
    message = f'A transcript of your conversation with {name}  is attached.'
    quickmail(f'Your conversation with {name} ({date}).',
              message=message,
              to_email=user_email,
              from_email=EMAIL,
              attach_paths=tmp_path)
    if cleanup: tmp_path.unlink()
    return True


def reset_app_state(end_conv=True, clear_queue=True, auto_punct=False,
                    attrs=('name', 'email')):
    """Reset some app-level attributes in `state`, `ask`, and `conv` objects.

    Parameters
    ----------
    end_conv: bool
        If True, end the conversation in our ConversationManager. Useful
        because we often check its current['persona'] attribute to determine if
        a conversation is underway.
    clear_queue: bool
        If True, clear Ask's queue of functions to call later.
    auto_punct: bool
        If True, use gpt-j to auto-punctuate/capitalize alexa's transcriptions
        before using them as prompts for a reply.
    attrs: Iterable[str]
        User attributes to retrieve if permission has been granted.
    """
    if end_conv:
        CONV.end_conversation()
    CONV.clear_default_kwargs(all_=True)
    if clear_queue:
        ask.func_clear()
    GPT.switch('banana' if ARGS.dev else 'openai')
    state.auto_punct = auto_punct
    for k, v in get_user_info(attrs).items():
        setattr(state, k, v)
        if k == 'name' and v:
            CONV.me = v
    # If we ever want explicitly tracked settings to include stop phrases,
    # this must remain after changing conv.me (see line above) since that
    # changes the stop phrases.
    state.init_settings(CONV, drop_fragment=True)
    # Note that in dev mode, we use banana backend which only has one model
    # (essentially equivalent to model=1).
    state.set('global', model=3)


@ask.launch
def launch():
    """Runs when user starts skill with command like 'Alexa, start Voice Chat'.
    """
    reset_app_state()
    question_txt = _choose_person_text()
    return question(f'Hi {state.name or "there"}! {question_txt}')\
        .reprompt('I didn\'t get that. Who would you like to speak to?')


def _choose_person_text(msg='Who would you like to speak to?'):
    # This is just a backup measure in case the user's next response gets sent
    # to delegate().
    ask.func_push(choose_person)
    return msg


@ask.intent('changeBackend')
def change_backend(backend=None):
    """Change the model backend (openai, gooseai, maybe others in the future)
    being used to generate responses.

    Sample Utterances
    -----------------
    "Lou, use gooseai backend."
    "Lou, change backend to openai."
    "Lou, please switch backend to banana."
    "Lou, switch to huggingface backend."
    """
    backend_name = backend or slot(request, 'backend', default='gooseai')\
        .replace(' ', '')
    msg = f'I\'ve switched your backend to {backend_name}.'
    try:
        GPT.switch(backend_name)
    except RuntimeError:
        msg = f'It sounded like you asked for backend ' \
              f'{backend_name or "no choice specified"}, but the only ' \
              'valid options are: "Open AI" and "Goose AI". You are ' \
              f'currently still using backend {GPT.current()}.'
    return _maybe_choose_person(msg)


@ask.intent('choosePerson')
def choose_person(person=None, **kwargs):
    """Allow the user to choose which person to talk to. If the user isn't
    recognized in their "contacts", we can autogenerate the persona for them
    if the person is relatively well known.

    Slots
    -----
    Person: str
        Name of a person to chat with.

    Sample Utterances
    -----------------
    "Harry Potter" (after being asked "Who would you like to speak to?")
    """
    person = person or kwargs.get('response') or slot(request, 'Person')
    # Handle case where conversation is already ongoing. This should have been
    # a reply - it just happened to consist of only a name.
    if CONV.is_active():
        return _reply(prompt=person)
    if person in NOBODY_UTTS:
        return statement('Goodbye.')
    # This handles our 2 alexa utterance collisions, "oh no" and "hush"
    # which alexa mistakenly maps to the choosePerson intent. If we're already
    # in a conversation, the above if clause handles it.
    if person.lower() in ('hush', 'oh no'):
        return _maybe_choose_person(
            'Sorry, I\'m confused.',
            choose_msg='You don\'t have a conversation in progress. I can '
                       'start one if you like - who do you want to speak to?'
        )
    # Case where name was originally unrecognized and had a match score >= .6
    # but less than .8, and user answered "No" when asked if the nearest
    # persona is who they meant.
    if not kwargs.get('choice', True):
        ask.func_push(_generate_person, person=kwargs['original'])
        return question('Ok. Would you like to create a new contact?')

    if person not in CONV:
        match, match_p = CONV.nearest_persona(person)
        ask.logger.info(f'Nearest matching persona: {match} (p={match_p:.3f})')
        if match_p < .6:
            ask.func_push(_generate_person, person=person)
            return question(
                f'I don\'t see anyone named {person} in your contacts. '
                'Would you like to create a new contact?'
            )
        elif match_p < .8:
            ask.func_push(choose_person, person=match, original=person)
            return question(
                f'I may have misheard but I don\'t see anyone named {person} '
                f'in your contacts, but I did see {match}. Is that who you '
                'meant?'
            )

        person = match
    CONV.start_conversation(person)
    state.on_conv_start(CONV)
    state.polly_voice = select_polly_voice(CONV.current)
    ask.func_clear()
    # large_img_url logic is a bit weird because standard_card checks if url
    # is not None, not if it's falsy. Missing urls often (always?) resolve to
    # empty strings in conv.name2meta so we need the None to be outside the
    # get() call.
    return question(f'I\'ve connected you with {person}.')\
        .standard_card(title=person,
                       large_image_url=CONV.current.get('img_url') or None)


def _generate_person(choice, **kwargs):
    if choice:
        try:
            CONV.add_persona(kwargs['person'].title())
            return choose_person(person=kwargs['person'])
        except Exception as e:
            ask.logger.error(f'Failed to generate {kwargs["person"]}. '
                             f'\nError: {e}')
            msg = f'I\'m sorry, I wasn\'t able to add {kwargs["person"]} ' \
                  f'as a contact.'
    else:
        # Case: user declines to auto-generate. Maybe they misspoke or changed
        # their mind.
        msg = f'Okay.'
    response = _maybe_choose_person(
        msg, choose_msg='Who would you like to speak to instead?'
    )
    # Otherwise we get the "back to your conversation..." message so this
    # reprompt wouldn't make sense.
    if not CONV.is_active():
        response = response.reprompt(
            'I didn\'t get that. Who would you like to speak to?'
        )
    return response


@ask.intent('changeModel')
def change_model(scope=None, model=None):
    """Change the model (gpt3 davinci, gpt3 curie, gpt-j, etc.) being used to
    generate responses.

    Sample Utterances
    -----------------
    "Lou, use model 0."
    "Lou, change model to 1."
    "Lou, switch to model 2."
    """
    scope = scope or slot(request, 'Scope', default='global')
    model = model or slot(request, 'Model')
    # This step might not be necessary anymore - slot used to allow both names
    # and numbers, which I think caused it to always be parsed as a string. It
    # should always be a number now.
    str2int = {
        'zero': 0,
        'one': 1,
        'two': 2,
        'three': 3,
        '0': 0,
        '1': 1,
        '2': 2,
        '3': 3
    }
    print('MODEL pre-conversion:', model, 'type:', type(model))
    model = str2int.get(model, model)
    print('MODEL post-conversion:', model, 'type:', type(model))
    msg = f'I\'ve switched your {scope} model to model {model}.'
    if isinstance(model, int):
        state.set(scope, model=model)
    else:
        msg = f'It sounded like you asked for model ' \
              f'{model or "no choice specified"}, but the only ' \
              'valid options are: 0, 1, 2, 3, J, and Neo. You are currently ' \
              f'still using model {model_type(state)}.'
    return _maybe_choose_person(msg)


@ask.intent('changeMaxLength')
def change_max_length(scope=None, length=None):
    """Change the max number of tokens in a generated response. The max is
    2048. There are roughly 1.33 tokens per word. I've set the default to
    50 tokens, which equates to roughly 2-3 sentences.

    Sample Utterances
    -----------------
    "Lou, change max length to 75."
    "Lou, set max length to 50."
    "Lou, set max tokens to 33."
    """
    error_msg = 'Please choose a number greater than zero and less than ' \
                'or equal to 2048. It sounded like you said "{}".'
    parse_error_msg = 'I didn\'t recognize that length value. ' \
                      + error_msg.partition('.')[0]

    scope = scope or slot(request, 'Scope', default='global')
    length = length or slot(request, 'Number')

    try:
        # First check if Alexa parsing failed (slots converts "?" to "").
        # This occurs for both decimals and non-numeric words.
        # Then check that user provided a valid value. Error messages are
        # different depending on the problem.
        assert length, parse_error_msg
        length = int(length)
        assert 0 < length < 2048, error_msg
    except (TypeError, AssertionError) as e:
        return question(str(e).format(length))

    state.set(scope, max_tokens=length)
    return _maybe_choose_person(
        f'I\'ve changed your max response length to {length}.'
    )


@ask.intent('changeTemperature')
def change_temperature(scope=None, temp=None):
    """Allow user to change model temperature. Lower values (near 0) are often
    better for formal or educational contexts, e.g. a science tutor. Choose
    a value in (0, 100] and we will scale it appropriately behind the scenes.

    Sample Utterances
    -----------------
    "Lou, change temperature to 1."
    "Lou, set temp to 90."
    "Lou, set temperature to 20."
    """
    # Alexa's speech to text makes parsing decimals kind of difficult, so we
    # ask the user to set temperature out of 100 and rescale it behind the
    # scenes. E.g. a value of 10 becomes 0.1, i.e. low levels of surprise.
    error_msg = 'Your conversation temperature must be an integer greater ' \
                'than zero and less than or equal to 100. It sounded like ' \
                'you said "{}".'
    parse_error_msg = 'I didn\'t recognize that temperature value. ' \
                      + error_msg

    scope = scope or slot(request, 'Scope', default='global')
    temp = temp or slot(request, 'Number')

    try:
        # First check if Alexa parsing failed (slots converts "?" to "").
        # This occurs for both decimals and non-numeric words.
        # Then check that user provided a valid value. Error messages are
        # different depending on the problem.
        assert temp, parse_error_msg
        temp = int(temp)
        assert 0 < temp <= 100, error_msg
    except (TypeError, AssertionError) as e:
        return question(str(e).format(temp))

    state.set(scope, temperature=temp / 100)
    return _maybe_choose_person(
        f'I\'ve adjusted your {scope}-level temperature to {temp} percent.'
    )


@ask.intent('enableAutoPunctuation')
def enable_punctuation():
    """
    Sample Utterances
    -----------------
    "Lou, please use auto punctuation."
    "Lou, enable automatic punctuation."
    "Lou, please turn on automatic punctuation."
    "Lou, turn on auto punctuation."
    """
    state.auto_punct = True
    return _maybe_choose_person('I\'ve enabled automatic punctuation.')


@ask.intent('disableAutoPunctuation')
def disable_punctuation():
    """
    Sample Utterances
    -----------------
    "Lou, disable auto punctuation."
    "Lou, please disable automatic punctuation."
    "Lou please stop using auto punctuation."
    "Lou, turn off automatic punctuation."
    """
    state.auto_punct = False
    return _maybe_choose_person('I\'ve disabled automatic punctuation.')


def _maybe_choose_person(
        msg='', choose_msg='Now, who would you like to speak to?',
        return_msg_fmt='Now, back to your call with {}.'
):
    """Check if a conversation is already in progress and if not, prompt the
    user to choose someone to talk to. Just a little convenience function to
    use after changing settings or the like.

    Parameters
    ----------
    msg: str
        Task-specific text unrelated to choosing a person.
    choose_msg: str
        Text that will only be appended to msg if a current conversation is not
        in progress.

    Returns
    -------
    str
    """
    assert msg or choose_msg, \
        'You must provide at least one of msg or choose_msg.'

    msg = msg.rstrip(' ') + ' '
    if CONV.is_active():
        name = CONV.process_name(CONV.current['persona'],
                                 inverse=True).split()[0]
        msg += return_msg_fmt.format(name)
    else:
        msg += _choose_person_text(choose_msg)
    return question(msg)


def _reply(prompt=None):
    """Generic conversation reply. I anticipate this endpoint making up the
    bulk of conversations.
    """
    prompt = prompt or slot(request, 'response', lower=False)
    if not prompt:
        return question('Did you say something? I didn\'t catch that.')
    # Set max tokens conservatively. Openai docs estimate n_tokens:n_words
    # ratio is roughly 1.33 on average.
    # Considered hardcoding in banana backend here (the current prompt already
    # uses the same size model by default) but in a tiny benchmark it was ~5s
    # per response which is slow enough to take a bit of the magic out of it.
    # My qualitative observation is that banana often *feels* faster than that
    # but I don't know if that's actually true.
    ask.logger.info('PROMPT: ' + prompt)
    if state.auto_punct:
        ask.logger.info('BEFORE PUNCTUATION: ' + prompt)
        prompt, _ = PROMPTER.query(task='punctuate_alexa',
                                   text=prompt, strip_output=True,
                                   max_tokens=2 * len(prompt.split()))
        prompt = prompt[0]
    ask.logger.info('BEFORE QUERY: ' + prompt)
    text, _ = CONV.query(prompt, **state)
    # Add custom accent/emotion audio.
    text, is_ssml = voice(text[0], CONV.current, polly_name=state.polly_voice,
                          select_voice=True, emo_pipe=EMO_PIPE)
    # Reprompt buys me some more time if I'm taking a long time to respond.
    # The reprompt is not included in the conversation transcript so it's
    # recommended that you respond to the initial reply rather than the generic
    # "I can see you're thinking hard."-esque reprompt.
    # Don't run emotion classifier on reprompt message.
    reprompt_msg, _ = voice(
        np.random.choice(REPROMPTS), CONV.current,
        polly_name=state.polly_voice, select_voice=True, emo_pipe=None
    )
    return custom_question(text, is_ssml)\
        .reprompt(reprompt_msg)


@ask.intent('delegate')
def delegate():
    """Delegate to the right function when no intent is detected.
    """
    func, kwargs = ask.func_pop()
    response = slot(request, 'response', lower=False)
    matches = infer_intent(response, UTT2META)
    ask.logger.info('\nInferred intent match scores:')
    ask.logger.info(matches)
    # Currently inferred intents take precedence over enqueued functions -
    # I'm still a bit wary of how much faith to place in the queue's
    # correctness (though the same could be said about inferred intents 😬).
    if matches['intent']:
        ask.logger.info(f'CALLING INFERRED INTENT: {matches["intent"]}')
        inferred_func = ask.intent2func(matches['intent'])
        return inferred_func(
            **{k.lower(): v for k, v in matches['slots'].items()}
        )
    # If we ask a user "Who do you want to speak to next?" and they
    # say noone/nobody, we should just quit. Feels more natural than responding
    # "quit".
    if response.lower() in NOBODY_UTTS and not CONV.is_active():
        return statement('Goodbye.')
    if not func:
        # No chained intents are in the queue so we assume this is just another
        # turn in the conversation.
        return _reply(response)
    return func(response=response, **kwargs)


@ask.intent('AMAZON.YesIntent')
def yes():
    """
    Sample Utterances
    -----------------
    "Yes."
    "Yes please."
    """
    func, kwargs = ask.func_pop()
    if not func:
        return _reply(prompt='Yes.')
    return func(choice=True, **kwargs)


@ask.intent('AMAZON.NoIntent')
def no():
    """
    Sample Utterances
    -----------------
    "No."
    "No thank you."
    """
    func, kwargs = ask.func_pop()
    if not func:
        return _reply(prompt='No.')
    return func(choice=False, **kwargs)


@ask.intent('AMAZON.StopIntent')
def stop():
    """This can only be used to exit the skill - if you say this during a
    conversation, Lou cannot ask if you want an emailed transcript or to
    immediately start a new conversation.

    Sample Utterances
    -----------------
    "Goodbye."
    """
    sent = False
    msg = 'Goodbye.'
    if CONV.is_active():
        # Tried delegating to _reply() here but alexa still ended the session
        # within a few seconds. Seems you're not allows to override this to
        # keep the skill active. I also tried returning end_chat() but that
        # failed similarly. Best we can do is try to email the user
        # automatically - we're not allowed to ask another question so I'd
        # rather send some unnecessary emails than automatically lose the
        # conversation.
        ask.logger.warning('StopIntent was called mid-conversation. This may '
                           'have been unintentional on the user\'s part.')
        if state.email:
            sent = send_transcript(CONV, state.email)
    if sent:
        msg += ' I\'ve sent you a transcript of your conversation.'
    return statement(msg)


@ask.intent('readContacts')
def read_contacts():
    """
    Sample Utterances
    -----------------
    "Lou, who are my contacts?"
    "Lou, please read me my contacts."
    "Lou, can you read me my contacts?"
    """
    msg = f'Here are all of your contacts: {", ".join(CONV.personas())}. '
    # If they're in the middle of a conversation, don't ask anything - just let
    # them get back to it.
    return _maybe_choose_person(msg)


@ask.intent('readSettings')
def read_settings():
    """Read the user their query settings.

    Sample Utterance
    ----------------
    "Lou, what are my settings?"
    "Lou, read me my settings."
    """
    strings = []
    for k, v in dict(state).items():
        if k == 'mock_func': continue
        if listlike(v):
            v = f'a list containing the following items: {v}'
        strings.append(f'{k.replace("_", " ")} is {v}')
        # Do this in for loop rather than after so model name is read right
        # after model number.
        if k == 'model':
            strings.append(f'model name is {GPT.engine(v)}')
    msg = f'Here are your settings: {"; ".join(strings)}. ' \
          f'Your api backend is {GPT.current()}. ' \
          f'You are {"" if state.auto_punct else "not"} using automatic '\
          'punctuation to improve transcription quality.'
    return _maybe_choose_person(msg)


@ask.intent('endChat')
def end_chat():
    """End conversation with the current person.

    Sample Utterances
    -----------------
    "Lou, hang up."
    "Lou, end chat."
    """
    # Only offer this option if user has chosen to share their email AND
    # the conversation is non-empty. No need to ask if user starts a
    # conversation and immediately quits (as I often need to do for dev
    # purposes).
    if state.email and CONV.user_turns:
        ask.func_push(_end_chat)
        return question('Would you like me to send you a transcript of your '
                        'conversation?')
    return _end_chat(False)


def _end_chat(choice=None, **kwargs):
    """End conversation and optionally send transcript to user.

    Parameters
    ----------
    choice: bool
        If True, email transcript to user.
    kwargs: any
        Not entirely sure if this is needed but my thinking is since end_chat
        calls ask.func_push on this, it might need to allow kwargs in case it's
        passed "response=response" by delegate intent.
    """
    if choice is None:
        choice = kwargs.get('response')
    if not isinstance(choice, bool):
        raise ValueError(f'Choice must be a bool, not {type(choice)}.')

    if choice:
        if send_transcript(CONV, state.email):
            msg = 'I\'ve emailed you a transcript of your conversation. '
        else:
            msg = 'Something went wrong and I wasn\'t able to send you ' \
                  'a transcript. Sorry about that.'
    else:
        msg = 'Okay.'

    CONV.end_conversation()
    state.on_conv_end()
    ask.func_clear()
    return question(
        msg + _choose_person_text(' Who would you like to speak to next?')
    ).reprompt('I didn\'t get that. Who would you like to speak to next?')


@ask.session_ended
def end_session():
    """Called when user exits the skill. Note: I tried adding a goodbye message
    but it looks like that's not supported.
    """
    # Occasionally I've seen conversations get cut off unexpectedly (perhaps
    # echo fails to pick up any audio or something). Sending transcript just to
    # be safe. Maybe in the future we can add the option to resume a
    # conversation.
    if CONV.is_active() and CONV.user_turns:
        send_transcript(CONV, state.email)
    return '{}', 200


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    # WARNING: in dev mode, each new conversation resets backend to banana.
    # Use literal_eval for type rather than bool - the latter oddly still seems
    # to parse inputs as strings.
    parser.add_argument(
        '--dev', default=False, type=ast.literal_eval,
        help='If True, start the app in dev mode (uses free backend by '
             'default).'
    )
    parser.add_argument(
        '--voice', default=True, type=ast.literal_eval,
        help='If True, use custom voices from Polly. If False, use the '
             'default Alexa voice. This is generally a worse experience '
             '(can\'t auto change gender, for instance) but it does allow '
             'us to use Polly\'s limited emotions (just "excited" or '
             '"sadness" at the moment).'
    )
    parser.add_argument(
        '--custom', default=True, type=ast.literal_eval,
        help='If True, include custom personas. If False, only include '
             'auto-generated personas.'
    )
    ARGS = parser.parse_args()
    CONV = ConversationManager(custom_names=[] if ARGS.custom else False,
                               load_qa_pipe=not ARGS.dev)
    PROMPTER = PromptManager(['punctuate_alexa'], verbose=False)
    UTT2META = load('data/alexa/utterance2meta.pkl')
    # Weird values/spellings here are mis-transcriptions I observed Alexa make.
    NOBODY_UTTS = {'knobody', 'nobody', 'noone', 'no one', 'no1', 'no 1'}
    REPROMPTS = [
        'I know, it\'s a lot to take in.',
        'I can see you\'re thinking hard.',
        'I know you like to mull things over.',
        'I can see the gears turning.',
        'I\'m not going anywhere. Take your time.'
    ]
    EMO_PIPE = None if ARGS.voice else pipeline(
        'text-classification',
        model='j-hartmann/emotion-english-distilroberta-base',
        return_all_scores=False
    )

    decorate_functions(debug_decorator)
    # Set false because otherwise weird things happen to app state in the
    # middle of a conversation. Tried calling reset_app_state() in this if
    # block but it seems to need to be called after run() so session is not
    # None, but we can't explicitly call it there because app.run call blocks.
    app.run(debug=False)
