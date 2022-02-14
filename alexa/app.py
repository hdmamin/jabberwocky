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

from htools import params, quickmail, save
from jabberwocky.openai_utils import ConversationManager
from config import EMAIL


logging.getLogger('flask_ask').setLevel(logging.DEBUG)
app = Flask(__name__)
ask = Ask(app, '/')
conv = ConversationManager(['Albert Einstein']) # TODO: load all personas?
app.logger.debug('>>> Loading globals') # TODO: rm


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


def intent(name, **ask_kwargs):
    """Decorator that replaces built-in ask.intent decorator. Only difference
    is it automatically maps lowercase function parameter names to uppercase
    slot names, since AWS seems to enforce that convention in the Alexa
    console.

    Parameters
    ----------
    name: str
        Intent name as defined in Alexa console.
    ask_kwargs: any
        Additional kwargs to pass to ask.intent().
    """
    def decorator(func):
        mapping = {k: k.title() for k in params(func)}

        @wraps(func)
        def wrapper(*args, **kwargs):
            # TODO: rm prints. For debugging only.
            app.logger.debug('Cur intent:', func.__name__)
            app.logger.debug('Prev intent:', session.attributes.get('prev_intent'))
            deco = ask.intent(name, mapping=mapping, **ask_kwargs)
            session.attributes['prev_intent'] = func.__name__
            return deco(func)(*args, **kwargs)
        return wrapper
    return decorator


# TODO rm
@app.route('/')
def home():
    """For debugging purposes.
    """
    app.logger.debug('>>> IN HOME')
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
    app.logger.debug('>>> IN LAUNCH')
    session.attributes['kwargs'] = conv._kwargs
    return question('Who would you like to speak to?')


@intent('choosePerson')
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


@intent('chooseModel')
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
        # conv._kwargs.update(model_i=int(model)) # TODO: rm
        session.attributes['kwargs']['model_i'] = int(model)
        return statement(f'I\'ve switched your service to model {model}.')
    else:
        # TODO: handle other model engines
        return statement(f'Model {model} is not yet implemented.')


@intent('changeMaxLength')
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


@intent('changeTemperature')
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


@intent('response')
def reply(response):
    """Generic conversation reply. I anticipate this endpoint making up the
    bulk of conversations.
    """
    text, _ = conv.query(response, **session.attributes['kwargs'])
    return question(text)


@intent('debug')
def debug(response):
    """For debugging purposes, simple endpoint that just repeats back what the
    user said last.
    """
    return question(f'I\'m in debug mode. You just said: {response}.')


@intent('AMAZON.FallbackIntent')
def fallback():
    # TODO: maybe direct every request here and use this as a delegator of
    # sorts?
    app.logger.debug('>>> FALLBACK INTENT')
    return question('Could you repeat that?')


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
    app.logger.debug(f'>>> MAIN: {conv.personas()}') # TODO: rm
    app.run(debug=True)
