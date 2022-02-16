"""
WIP: trying Flask-Ask since I'm not sure how to run the lambda_function locally
of if that's even possible. App relies on local filesystem a lot at the moment
so it might be more convenient to run locally with ngrok anyway.
"""

from datetime import datetime
from functools import wraps
import logging
from pathlib import Path

from flask import Flask
from flask_ask import Ask, statement, question, session, context, request
import requests

from htools import params, quickmail, save, MultiLogger, Callback, callbacks
from jabberwocky.openai_utils import ConversationManager
from config import EMAIL


class IntentCallback(Callback):
    # TODO: docs

    def __init__(self, ask):
        self.ask = ask

    def setup(self, func):
        pass

    def on_begin(self, func, inputs, output=None):
        self.ask.logger.info('Cur intent: ' + self.ask.intent_name(func))
        self.ask.logger.info(
            'Prev intent: ' + session.attributes.get('prev_intent', '<NONE>')
        )

    def on_end(self, func, inputs, output=None):
        session.attributes['prev_intent'] = self.ask.intent_name(func)


class CustomAsk(Ask):
    # TODO: docs

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = MultiLogger('alexa/app.log', fmode='a')
        self._callbacks = callbacks([IntentCallback(self)])
        self._func2name = {}

    def intent_name(self, func):
        return self._func2name[func.__name__]

    def attach_callbacks(self, func):
        return self._callbacks(func)

    def intent(self, name, **ask_kwargs):
        def decorator(func):
            func = self.attach_callbacks(func)
            self._func2name[func.__name__] = name
            mapping = {k: k.title() for k in params(func)}
            self._intent_view_funcs[name] = func
            self._intent_mappings[name] = {**mapping,
                                           **ask_kwargs.get('mapping', {})}
            self._intent_converts[name] = ask_kwargs.get('convert', {})
            self._intent_defaults[name] = ask_kwargs.get('default', {})

            @wraps(func)
            def wrapper(*args, **kwargs):
                self._flask_view_func(*args, **kwargs)
            return func
        return decorator


logging.getLogger('flask_ask').setLevel(logging.DEBUG)
app = Flask(__name__)
ask = CustomAsk(app, '/')
conv = ConversationManager(['Albert Einstein']) # TODO: load all personas?


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
    if not (token and endpoint):
        return ''
    headers = {'Authorization': f'Bearer: {token}'}
    r = requests.get(f'{endpoint}/v2/accounts/~current/settings/Profile.email',
                     headers=headers)
    if r.status_code != 200:
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


@app.route('/health')
def health():
    """For debugging purposes (lets us check that the app is accessible).
    """
    print('IN HEALTH')
    return 'Jabberwocky is running.'


@ask.launch
def launch():
    """Runs when user starts skill with command like 'Alexa, start Voice Chat'.
    """
    app.logger.info('>>> IN LAUNCH')
    session.attributes['kwargs'] = conv._kwargs
    return question('Who would you like to speak to?')


@ask.intent('choosePerson')
def choose_person(person):
    """Allow the user to choose which person to talk to. If the user isn't
    recognized in their "contacts", we can autogenerate the persona for them
    if the person is relatively well known.

    Parameters
    ----------
    person: str
    """
    if person not in conv:
        # TODO: new endpoint needed to handle answer to this case?
        return question(f'I don\'t see anyone named {person} in your '
                        f'contacts. Would you like to create a new contact?')

    conv.start_conversation(person)
    return question(f'I\'ve connected you with {person}.')


@ask.intent('chooseModel')
def choose_model(model):
    """Change the model (gpt3 davinci, gpt3 curie, gpt-j, etc.) being used to
    generate responses.

    Parameters
    ----------
    model: str
    """
    if model is None:
        return statement('I didn\'t recognize that model type. You\'re '
                         f'still using {conv._kwargs["model_i"]}')
    if model.isdigit():
        session.attributes['kwargs']['model_i'] = int(model)
        return statement(f'I\'ve switched your service to model {model}.')
    else:
        # TODO: handle other model engines
        return statement(f'Model {model} is not yet implemented.')


@ask.intent('changeMaxLength')
def change_max_length(length):
    """Change the max number of tokens in a generated response. The max is
    2048. There are roughly 1.33 tokens per word. I've set the default to
    50 tokens, which equates to roughly 2-3 sentences.
    """
    try:
        length = int(length)
        assert 0 < length < 2048
    except (TypeError, AssertionError) as e:
        error_msg = 'Please choose a number greater than zero and less than ' \
                    'or equal to 2048.'
        if isinstance(e, TypeError):
            error_msg = 'I didn\'t recognize that value.' + error_msg
        return question(error_msg)

    session.attributes['kwargs']['max_tokens'] = length
    return question(f'Choose length {length}.')


@ask.intent('changeTemperature')
def change_temperature(temperature):
    """Allow user to change model temperature. Lower values (near 0) are often
    better for formal or educational contexts, e.g. a science tutor.
    """
    error_msg = 'Please choose a number greater than zero and ' \
                'less than or equal to one.'
    try:
        temperature = float(temperature)
        assert 0 < temperature <= 1
    except (TypeError, AssertionError) as e:
        if isinstance(e, TypeError):
            error_msg = ('I didn\'t recognize that temperature value. ' +
                         error_msg)
        return question(error_msg)

    session.attributes['kwargs']['temperature'] = temperature
    return question(f'I\'ve adjusted your temperature to {temperature}.')


@ask.intent('response')
def reply(response):
    """Generic conversation reply. I anticipate this endpoint making up the
    bulk of conversations.
    """
    text, _ = conv.query(response, **session.attributes['kwargs'])
    return question(text)


@ask.intent('debug')
def debug(response):
    """For debugging purposes, simple endpoint that just repeats back what the
    user said last.
    """
    return question(f'I\'m in debug mode. You just said: {response}.')


# TODO
@ask.intent('AMAZON.FallbackIntent')
def fallback():
    # TODO: maybe direct every request here and use this as a delegator of
    # sorts?
    return question('Could you repeat that?')


@ask.intent('readContacts')
def read_contacts():
    msg = f'Here are all of your contacts: {",".join(conv.personas)}. ' \
          f'Now, who would you like to speak to?'
    return question(msg)


@ask.session_ended
def end_session():
    return '{}', 200


def exit_():
    saved = False
    if session.attributes.get('should_end') and \
                session.attributes.get('should_save'):
            # TODO: have user configure this at some point earlier?
            saved = send_transcript(conv, session.attributes['email'])
    return saved


if __name__ == '__main__':
    app.logger.info(f'>>> MAIN: {conv.personas()}') # TODO: rm
    app.run(debug=True)
