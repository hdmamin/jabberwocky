"""
WIP: trying Flask-Ask since I'm not sure how to run the lambda_function locally
of if that's even possible. App relies on local filesystem a lot at the moment
so it might be more convenient to run locally with ngrok anyway.
"""

from datetime import datetime
from functools import wraps, partial
import logging
from pathlib import Path
import sys

from flask import Flask, request
from flask_ask import Ask, statement, question, session, context
import requests

from config import EMAIL
from htools import params, quickmail, save, MultiLogger, Callback, callbacks, \
    listlike
from jabberwocky.openai_utils import ConversationManager, query_gpt3, \
    query_gpt_j, query_gpt_neo
from jabberwocky.utils import load_huggingface_api_key
from utils import slot, word2int, Settings, model_type


class IntentCallback(Callback):
    # TODO: docs
    # TODO: maybe move to utils? But relies on state var. Could pass that to
    # init maybe?

    def __init__(self, ask):
        self.ask = ask

    def setup(self, func):
        pass

    def on_begin(self, func, inputs, output=None):
        self.ask.logger.info('\nCur intent: ' + self.ask.intent_name(func))
        self.ask.logger.info(f'Prev intent: {state.prev_intent}\n')
        self.ask.logger.info(state)

    def on_end(self, func, inputs, output=None):
        state.prev_intent = self.ask.intent_name(func)


class CustomAsk(Ask):
    """Slightly customized version of flask-ask's Ask object. See `intent`
    method for a summary of main changes.

    # TODO: move to utils? Depends on if we can move IntentCallback (see its
    docstring).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Unlike flask app.logger, this writes to both stdout and a log file.
        self.logger = MultiLogger('alexa/app.log', fmode='a')
        # Decorator that we use on each intent endpoint.
        self._callbacks = callbacks([IntentCallback(self)])
        self._func2intent = {}
        self._intent2funcname = {}

    def followup_func(self, prev_intent):
        """Some endpoints ask a followup question and we need to direct their
        response to a different function. This function maps from an intent
        name to the function (not the name, the actual function) that should
        be called next.

        Parameters
        ----------
        prev_intent: str

        Returns
        -------
        FunctionType
        """
        prev_funcname = self._intent2funcname[prev_intent]
        return getattr(sys.modules['__main__'], f'_{prev_funcname}')

    def intent_name(self, func) -> str:
        """Given a flask endpoint function, return the name of the intent
        associated with it.
        """
        return self._func2intent[func.__name__]

    def attach_callbacks(self, func):
        """Prettier way to wrap an intent function with callbacks. This adds
        logging showing what the current and previous intent are/were and also
        updates session state to allow for this kind of tracking.

        Returns
        -------
        FunctionType: A decorated intent endpoint function.
        """
        return self._callbacks(func)

    def intent(self, name, **ask_kwargs):
        """My version of ask.intent decorator, overriding the default
        implementation. Changes:
        - Automatically map map slot names from title case to lowercase. AWS
        console seems to enforce some level of capitalization that I'd prefer
        not to use in all my python code.
        - Populate a dict mapping endpoint function -> intent name. These are
        usually similar but not identical (often just a matter of
        capitalization but not always).

        Parameters
        ----------
        name: str
            Name of intent.
        ask_kwargs: dict(s)
            Additional kwargs for ask.intent (effectively - we don't explicitly
            call it, but rather reproduce its functionality below). E.g.
            `mapping`, `convert`, or `default`.
        """
        def decorator(func):
            func = self.attach_callbacks(func)
            self._func2intent[func.__name__] = name
            self._intent2funcname[name] = func.__name__
            mapping = {k: k.title() for k in params(func)}
            self._intent_view_funcs[name] = func
            self._intent_mappings[name] = {**mapping,
                                           **ask_kwargs.get('mapping', {})}
            self._intent_converts[name] = ask_kwargs.get('convert', {})
            self._intent_defaults[name] = ask_kwargs.get('default', {})

            @wraps(func)
            def wrapper(*args, **kwargs):
                """This looks useless - we don't return wrapper and we never
                seemed to reach this part of the code when I added logging -
                but it's in the built-in implementation of `intent` in
                flask-ask so I don't know what other library logic might rely
                on it. Just keep it.
                """
                self._flask_view_func(*args, **kwargs)
            return func
        return decorator


logging.getLogger('flask_ask').setLevel(logging.DEBUG)
app = Flask(__name__)
# Necessary to make session accessible outside endpoint functions.
app.app_context().push()
ask = CustomAsk(app, '/')
conv = ConversationManager(['Albert Einstein']) # TODO: load all personas?
state = Settings()


def get_user_email():
    """Get user's email using Amazon-provided API. Obviously only works if
    they've approved access to this information.

    Returns
    -------
    str: User email if user gave permission and everything worked, empty string
    otherwise.
    """
    system = context.System
    token = system.get("apiAccessToken")
    endpoint = system.get('apiEndpoint')
    print(f'token={token}', f'endpoint={endpoint}')  # TODO rm
    if not (token and endpoint):
        return ''
    headers = {'Authorization': f'Bearer {token}'}
    r = requests.get(f'{endpoint}/v2/accounts/~current/settings/Profile.email',
                     headers=headers)
    if r.status_code != 200:
        print('status code', r.status_code, r.reason)
        return ''
    return r.json().get('emailAddress', '')


def send_transcript(conv, user_email=''):
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

    Returns
    -------
    bool: True if email was sent successfully, False otherwise (e.g. if no
    conversation has taken place yet or if the user has not provided us with
    their email).
    """
    if not conv.user_turns:
        return False
    user_email = user_email or get_user_email()
    if not user_email:
        return False
    date = datetime.today().strftime('%m/%d/%Y')
    tmp_path = Path('/tmp/jabberwocky-transcript.txt')
    save(conv.full_conversation(), tmp_path)
    message = 'A transcript of your conversation with ' \
              f'{conv.current_persona}  is attached.'
    quickmail(f'Your conversation with {conv.current_persona} ({date}).',
              message=message,
              to_email=user_email,
              from_email=EMAIL,
              attach_path=tmp_path)
    tmp_path.unlink()
    return True


# TODO rm
@app.route('/')
def home():
    """For debugging purposes.
    """
    app.logger.info('>>> IN HOME')
    return 'home'


@ask.launch
def launch():
    """Runs when user starts skill with command like 'Alexa, start Voice Chat'.
    """
    state.set('global', **conv._kwargs)
    # TODO: might want to change this eventually, but for now use free model
    # by default.
    state.set('global', mock_func=query_gpt_j)
    state.email = get_user_email()
    print('LAUNCH email', state.email) # TODO rm
    return question('Welcome to Quick Chat. Who would you like to speak to?')


@ask.intent('debug')
def debug(response):
    """For debugging purposes, simple endpoint that just repeats back what the
    user said last.
    """
    return question(f'I\'m in debug mode. You just said: {response}.')


@ask.intent('choosePerson')
def choose_person():
    """Allow the user to choose which person to talk to. If the user isn't
    recognized in their "contacts", we can autogenerate the persona for them
    if the person is relatively well known.

    Slots
    -----
    Person: str
        Name of a person to chat with.
    """
    person = slot(request, 'Person')
    if conv.current_persona:
        # TODO: try to get full user text and call _reply().
        # Currently this cuts off the first since it thinks the pattern is
        # {any one word} {name}.
        return question(person)

    print('PERSON', person) # TODO rm
    if person not in conv:
        state.kwargs = {'person': person}
        return question(f'I don\'t see anyone named {person} in your '
                        f'contacts. Would you like to create a new contact?')

    conv.start_conversation(person)
    return question(f'I\'ve connected you with {person}.')


def _choose_person(choice, **kwargs):
    if choice:
        try:
            conv.start_conversation(kwargs['person'],
                                    download_if_necessary=True)
            msg = f'I\'ve connected you with {kwargs["person"]}.'
        except Exception as e:
            ask.logger.error(f'Failed to generate {kwargs["person"]}.\n{e}')
            msg = f'I\'m sorry, I wasn\'t able to add {kwargs["person"]} ' \
                  f'as a contact. Who would you like to speak to instead?'
    else:
        # Case: user declines to auto-generate. Maybe they misspoke or changed
        # their mind.
        msg = 'Okay. Who would you like to speak too, then?'
    return question(msg)


@ask.intent('changeModel')
def change_model():
    """Change the model (gpt3 davinci, gpt3 curie, gpt-j, etc.) being used to
    generate responses.

    Parameters
    ----------
    model: str
    """
    scope = slot(request, 'Scope', default='global')
    model = slot(request, 'Model')
    # Conversion is not automatic here because we're not using a built-in
    # AMAZON.Number slot (because some values aren't numbers).
    str2int = {
        'zero': 0,
        'one': 1,
        'two': 2,
        'three': 3
    }
    model = str2int.get(model, model)
    print('MODEL', model)
    msg = f'I\'ve switched your {scope} backend to model {model}.'
    if isinstance(model, int):
        state.set(scope, model_i=model)
    elif model == 'j':
        state.set(scope, mock_func=query_gpt_j)
    elif model == 'neo':
        state.set(scope, mock_func=partial(query_gpt_neo, api_key=HF_API_KEY))
    else:
        msg = f'It sounded like you asked for model ' \
              f'{model or "no choice specified"}, but the only ' \
              'valid options are: 0, 1, 2, 3, J, and Neo. You are currently ' \
              f'still using model {model_type(state)}.'
    return question(msg)


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
    length = slot(request, 'MaxLength')

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
    return question(f'I\'ve changed your max response length to {length}.')


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
    return question(f'I\'ve adjusted your {scope}-level temperature to {temp} '
                    f'percent.')


@ask.intent('reply')
def reply():
    """Generic conversation reply. I anticipate this endpoint making up the
    bulk of conversations.
    """
    prompt = slot(request, 'response', lower=False)
    print('REPLY prompt', prompt) # TODO rm
    if not prompt:
        return question('Did you say something? I didn\'t catch that.')
    _, text = conv.query(prompt, **state)
    return question(text)


# TODO
@ask.intent('AMAZON.FallbackIntent')
def fallback():
    # TODO: maybe direct every request here and use this as a delegator of
    # sorts? Or should I make the reply function correspond to the
    # FallbackIntent?
    return question('Could you repeat that?')


@ask.intent('AMAZON.YesIntent')
def yes():
    # TODO: may need to handle case where user says Yes as a reply, not as a
    # YesIntent.
    # TODO: action depends on prev intent.
    func = ask.followup_func(state.prev_intent)
    print('Yes (placeholder).')
    # TODO: might need to pass kwargs in, not just True/False, if some
    # followups take more complex/different slots.
    return func(choice=True, **state.kwargs)


@ask.intent('AMAZON.NoIntent')
def no():
    # TODO: action depends on prev intent.
    func = ask.followup_func(state.prev_intent)
    print('No (placeholder).')
    # TODO: might need to pass kwargs in, not just True/False, if some
    # followups take more complex/different slots.
    return func(choice=False, **state.kwargs)


@ask.intent('readContacts')
def read_contacts():
    msg = f'Here are all of your contacts: {", ".join(conv.personas())}. ' \
          f'Now, who would you like to speak to?'
    return question(msg)


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
    return question(f'Here are your settings: {"; ".join(strings)}.')


@ask.intent('endChat')
def end_chat():
    """Read the user their query settings.

    Sample utterance:
    "Lou, end chat."
    "Lou, hang up."
    """
    return question('Would you like me to send you a transcript of your '
                    'conversation?')


@ask.session_ended
def end_session():
    """Called when user exits the skill."""
    return '{}', 200


def _end_chat(choice):
    """End conversation and optionally send transcript to user.

    Parameters
    ----------
    choice: bool
        If True, email transcript to user.
    """
    if choice:
        if state.email:
            sent = send_transcript(conv, state.email)
            if sent:
                msg = 'I\'ve emailed you a transcript of your conversation. '
            else:
                msg = 'Something went wrong and I wasn\'t able to send you ' \
                      'a transcript. Sorry about that.'
        # Defaults to empty str when no email is provided.
        else:
            msg = 'I don\'t have your email on file.'
    else:
        msg = 'Okay.'
    conv.end_conversation()
    return question(msg + ' Would you like to talk to someone else?')


if __name__ == '__main__':
    HF_API_KEY = load_huggingface_api_key()
    app.logger.info(f'>>> MAIN: {conv.personas()}') # TODO: rm
    app.run(debug=True)
