"""
WIP: trying Flask-Ask since I'm not sure how to run the lambda_function locally
of if that's even possible. App relies on local filesystem a lot at the moment
so it might be more convenient to run locally with ngrok anyway.
"""

from datetime import datetime
from functools import partial
import logging
from pathlib import Path

from flask import Flask, request
from flask_ask import question, context, statement
import requests

from config import EMAIL, HF_API_KEY
from htools import quickmail, save, tolist, listlike, decorate_functions,\
    debug as debug_decorator
from jabberwocky.openai_utils import ConversationManager, query_gpt_j,\
    query_gpt_neo, PromptManager, BackendSelector
from utils import slot, Settings, model_type, CustomAsk


# Define these before functions since endpoints use ask method as decorators.
# Unclutter terminal output because it's hard to debug otherwise.
logging.getLogger('flask_ask').setLevel(logging.WARNING)
app = Flask(__name__)
# Necessary to make session accessible outside endpoint functions.
app.app_context().push()
state = Settings()
ask = CustomAsk(state, app, '/')


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
    # ask.logger.info(f'token={token}', f'endpoint={endpoint}')  # TODO rm
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


# TODO: do I need to ask this somewhere? I enabled permission for MY account
# in alexa app and alexa site already but this is probably needed if I want to
# let someone else use it. Right now, the card is displayed after returning
# this (good) but if I respond Yes or No I just get an "audio response" (beep)
# from alexa. Unsure what this means.
@ask.intent('emailMe')
def tmp_email_me():
    return statement(
        'Do you mind if I access your name and email address? This will let '
        'me send you transcripts of your conversations.'
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
    if not conv.user_turns:
        return False
    user_email = user_email or get_user_info('email')['email']
    if not user_email:
        return False
    date = datetime.today().strftime('%m/%d/%Y')
    datetime_ = datetime.today().strftime('%Y.%m.%d__%H.%M.%S')
    tmp_path = Path(
        f'alexa/conversations/{conv.current_persona}__{datetime_}.txt'
    )
    save(conv.full_conversation(), tmp_path)
    message = 'A transcript of your conversation with ' \
              f'{conv.current_persona}  is attached.'
    quickmail(f'Your conversation with {conv.current_persona} ({date}).',
              message=message,
              to_email=user_email,
              from_email=EMAIL,
              attach_paths=tmp_path)
    if cleanup: tmp_path.unlink()
    return True


def reset_app_state(end_conv=True, clear_queue=True, backend_='j',
                    auto_punct=True, attrs=('name', 'email')):
    """Reset some app-level attributes in `state`, `ask`, and `conv` objects.
    This does NOT reset gpt3 query kwargs aside from replacing all gpt3 calls
    with gptj calls by default.

    Parameters
    ----------
    end_conv: bool
        If True, end the conversation in our ConversationManager. Useful
        because we often check its current_persona attribute to determine if
        a conversation is underway.
    clear_queue: bool
        If True, clear Ask's queue of functions to call later.
    use_gpt_j: bool
        If true, replace gpt3 calls with free gpt-j calls.
    auto_punct: bool
        If True, use gpt-j to auto-punctuate/capitalize alexa's transcriptions
        before using them as prompts for a reply.
    attrs: Iterable[str]
        User attributes to retrieve if permission has been granted.
    """
    if end_conv:
        conv.end_conversation()
    if clear_queue:
        ask.func_clear()
    state.kwargs = {}
    # TODO: might want to change this eventually, but for now use free model
    # by default.
    if backend_ == 'j':
        state.set('global', mock_func=query_gpt_j)
    if backend_ in ('gooseai', 'openai'):
        backend.switch(backend_)
    state.auto_punct = auto_punct
    for k, v in get_user_info(attrs).items():
        setattr(state, k, v)
        if k == 'name' and v:
            conv.me = v


@ask.launch
def launch():
    """Runs when user starts skill with command like 'Alexa, start Voice Chat'.
    """
    reset_app_state()
    question_txt = _choose_person_text()
    return question(f'Hi {state.name or "there"}! Welcome to Quick Chat. '
                    f'{question_txt}')\
        .reprompt('I didn\'t get that. Who would you like to speak to next?')


def _choose_person_text(msg='Who would you like to speak to?'):
    # This is just a backup measure in case the user's next response gets sent
    # to delegate().
    ask.func_push(choose_person)
    return msg


@ask.intent('choosePerson')
def choose_person(**kwargs):
    """Allow the user to choose which person to talk to. If the user isn't
    recognized in their "contacts", we can autogenerate the persona for them
    if the person is relatively well known.

    Slots
    -----
    Person: str
        Name of a person to chat with.
    """
    # Don't put slot call as the default value in get because that would cause
    # it to be executed before checking if kwargs contains the value we want.
    # When the name is passed in, there will likely be no slots and that call
    # would raise an error.
    person = kwargs.get('response') or slot(request, 'Person')
    # Handle case where conversation is already ongoing. This should have been
    # a reply - it just happened to consist of only a name.
    if conv.current_persona:
        return _reply(prompt=person)

    if person not in conv:
        matches = [p for p in map(str.lower, conv.personas())
                   if person == p.split()[-1]]
        # Allows us to just say "Einstein" rather than "Albert Einstein". If
        # we have multiple matches, don't try to guess (e.g. "Armstrong" could
        # refer to either Neil Armstrong or Louis Armstrong). Considered fuzzy
        # matching but I don't think that's desirable here.
        if len(matches) == 1:
            person = matches[0]
        else:
            state.kwargs = {'person': person}
            ask.func_push(_generate_person)
            return question(
                f'I don\'t see anyone named {person} in your contacts. '
                'Would you like to create a new contact?'
            )

    conv.start_conversation(person)
    ask.func_clear()
    return question(f'I\'ve connected you with {person}.')


def _generate_person(choice, **kwargs):
    if choice:
        try:
            conv.add_persona(kwargs['person'].title())
            return choose_person(response=kwargs['person'])
        except Exception as e:
            ask.logger.error(f'Failed to generate {kwargs["person"]}. '
                             f'\nError: {e}')
            msg = f'I\'m sorry, I wasn\'t able to add {kwargs["person"]} ' \
                  f'as a contact. Who would you like to speak to instead?'
    else:
        # Case: user declines to auto-generate. Maybe they misspoke or changed
        # their mind.
        msg = f'Okay. {_choose_person_text()}'
    return question(msg) \
        .reprompt('I didn\'t get that. Who would you like to speak to next?')


@ask.intent('changeModel')
def change_model():
    """Change the model (gpt3 davinci, gpt3 curie, gpt-j, etc.) being used to
    generate responses.

    Sample Utterances
    -----------------
    "Lou, use model j."
    "Lou, change global model to 2."
    """
    scope = slot(request, 'Scope', default='global')
    model = slot(request, 'Model')
    # Conversion is not always automatic here, I think because we're not using
    # a built-in AMAZON.Number slot (because some values aren't numbers).
    # Or maybe it's that saying "two", typing "two", or typing "2" give
    # different transcriptions - unsure.
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
    model = str2int.get(model, model)
    print('MODEL', model, 'type:', type(model))
    msg = f'I\'ve switched your {scope} backend to model {model}.'
    if isinstance(model, int):
        state.set(scope, model_i=model)
        state.set(scope, mock_func=None)
    elif model == 'j':
        state.set(scope, mock_func=query_gpt_j)
    elif model == 'neo':
        state.set(scope, mock_func=partial(query_gpt_neo, api_key=HF_API_KEY))
    else:
        msg = f'It sounded like you asked for model ' \
              f'{model or "no choice specified"}, but the only ' \
              'valid options are: 0, 1, 2, 3, J, and Neo. You are currently ' \
              f'still using model {model_type(state)}.'
    return _maybe_choose_person(msg)


@ask.intent('changeMaxLength')
def change_max_length():
    """Change the max number of tokens in a generated response. The max is
    2048. There are roughly 1.33 tokens per word. I've set the default to
    50 tokens, which equates to roughly 2-3 sentences.
    """
    error_msg = 'Please choose a number greater than zero and less than ' \
                'or equal to 2048. It sounded like you said "{}".'
    parse_error_msg = 'I didn\'t recognize that length value. ' \
                      + error_msg.partition('.')[0]

    scope = slot(request, 'Scope', default='global')
    length = slot(request, 'Number')

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
def change_temperature():
    """Allow user to change model temperature. Lower values (near 0) are often
    better for formal or educational contexts, e.g. a science tutor.
    """
    # Alexa's speech to text makes parsing decimals kind of difficult, so we
    # ask the user to set temperature out of 100 and rescale it behind the
    # scenes. E.g. a value of 10 becomes 0.1, i.e. low levels of surprise.
    error_msg = 'Your conversation temperature must be an integer greater ' \
                'than zero and less than or equal to 100. It sounded like ' \
                'you said "{}".'
    parse_error_msg = 'I didn\'t recognize that temperature value. ' \
                      + error_msg

    scope = slot(request, 'Scope', default='global')
    temp = slot(request, 'Number')

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
    state.auto_punct = True
    return _maybe_choose_person('I\'ve enabled automatic punctuation.')


@ask.intent('disableAutoPunctuation')
def disable_punctuation():
    state.auto_punct = False
    return _maybe_choose_person('I\'ve disabled automatic punctuation.')


def _maybe_choose_person(msg='',
                         choose_msg='Now, who would you like to speak to?'):
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

    if not conv.current_persona:
        msg = msg.rstrip(' ') + ' ' + _choose_person_text(choose_msg)
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
    # TODO: maybe add setting to make punctuation optional? Prob slows things
    # down significantly, but potentially could improve completions a lot.
    ask.logger.info('BEFORE PUNCTUATION: ' + prompt)
    _, prompt = gpt.query(task='punctuate_alexa', text=prompt,
                          mock_func=query_gpt_j, strip_output=True,
                          max_tokens=2 * len(prompt.split()))
    ask.logger.info('AFTER PUNCTUATION: ' + prompt)
    _, text = conv.query(prompt, **state)
    return question(text)


@ask.intent('delegate')
def delegate():
    """Delegate to the right function when no intent is detected.
    """
    func = ask.func_pop()
    response = slot(request, 'response', lower=False)
    if not func:
        # No chained intents are in the queue so we assume this is just another
        # turn in the conversation.
        return _reply(response)
    return func(response=response, **state.kwargs or {})


@ask.intent('AMAZON.YesIntent')
def yes():
    func = ask.func_pop()
    if not func:
        # TODO: better validation to ensure the user meant to send this as a
        # reply, not just bc the queue is empty.
        # TODO: what to do when nothing in queue? Maybe send to reply()?
        return _reply(prompt='Yes.')
    return func(choice=True, **state.kwargs)


@ask.intent('AMAZON.NoIntent')
def no():
    func = ask.func_pop()
    if not func:
        # TODO: better validation to ensure the user meant to send this as a
        # reply, not just bc the queue is empty.
        # TODO: what to do when nothing in queue? Maybe send to reply()?
        # TODO: what to do when nothing in queue? Maybe send to reply()?
        return _reply(prompt='No.')
    return func(choice=False, **state.kwargs)


@ask.intent('readContacts')
def read_contacts():
    """
    Sample utterance:
    "Lou, read me my contacts."
    "Lou, who are my contacts?"
    """
    msg = f'Here are all of your contacts: {", ".join(conv.personas())}. '
    # If they're in the middle of a conversation, don't ask anything - just let
    # them get back to it.
    return _maybe_choose_person(msg)


@ask.intent('readSettings')
def read_settings():
    """Read the user their query settings.

    Sample utterance:
    "Lou, what are my settings?"
    """
    strings = []
    for k, v in dict(state).items():
        if k == 'mock_func': continue
        if listlike(v):
            v = f'a list containing the following items: {v}'
        strings.append(f'{k.replace("_", " ")} is {v}')
    msg = f'Here are your settings: {"; ".join(strings)}. You are ' \
          f'{"" if state.auto_punct else "not"} using automatic punctuation ' \
          f'to improve transcription quality.'
    return _maybe_choose_person(msg)


@ask.intent('endChat')
def end_chat():
    """End conversation with the current person.

    Sample utterance:
    "Lou, end chat."
    "Lou, hang up."
    """
    # Only offer this option if user has chosen to share their email.
    if state.email:
        ask.func_push(_end_chat)
        return question('Would you like me to send you a transcript of your '
                        'conversation?')
    return _end_chat(False)


def _end_chat(choice):
    """End conversation and optionally send transcript to user.

    Parameters
    ----------
    choice: bool
        If True, email transcript to user.
    """
    if choice:
        if send_transcript(conv, state.email):
            msg = 'I\'ve emailed you a transcript of your conversation. '
        else:
            msg = 'Something went wrong and I wasn\'t able to send you ' \
                  'a transcript. Sorry about that.'
    else:
        msg = 'Okay.'

    # TODO: prob rm. Just wait to see what user consent for email looks like
    # (IE is there any chane someone could forget to give consent and not
    # realize it until now? Prob should design experience so that
    # doesn't happen).
    # if choice:
    #     if state.email:
    #         sent = send_transcript(conv, state.email)
    #         if sent:
    #             msg = 'I\'ve emailed you a transcript of your conversation. '
    #         else:
    #             msg = 'Something went wrong and I wasn\'t able to send you ' \
    #                   'a transcript. Sorry about that.'
    #     # Defaults to empty str when no email is provided.
    #     else:
    #         msg = 'Sorry, I can\'t because I don\'t have your email on file.'
    # else:
    #     msg = 'Okay.'
    conv.end_conversation()
    ask.func_clear()
    return question(
        msg + _choose_person_text(' Who would you like to speak to next?')
    ).reprompt('I didn\'t get that. Who would you like to speak to next?')


@ask.session_ended
def end_session():
    """Called when user exits the skill. Note: I tried adding a goodbye message
    but it looks like that's not supported.
    """
    return '{}', 200


if __name__ == '__main__':
    conv = ConversationManager(['Albert Einstein'])  # TODO: load all personas?
    gpt = PromptManager(['punctuate_alexa'], verbose=False)
    backend = BackendSelector()
    decorate_functions(debug_decorator)
    # Set false because otherwise weird things happen to app state in the
    # middle of a conversation. Tried calling reset_app_state() in this if
    # block but it seems to need to be called after run() so session is not
    # None, but we can't explicitly call it there because app.run call blocks.
    app.run(debug=False)
