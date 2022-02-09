"""
WIP: trying Flask-Ask since I'm not sure how to run the lambda_function locally
of if that's even possible. App relies on local filesystem a lot at the moment
so it might be more convenient to run locally with ngrok anyway.
"""

from datetime import datetime
from functools import wraps
import logging

from flask import Flask
from flask_ask import Ask, statement, question, session, context, request
import requests

from htools import params, quickmail
from jabberwocky.openai_utils import ConversationManager
from config import EMAIL


logging.getLogger('flask_ask').setLevel(logging.DEBUG)
app = Flask(__name__)
ask = Ask(app, '/')
conv = ConversationManager(['Albert Einstein']) # TODO: load all personas?
app.logger.warning('>>> Loading globals') # TODO: rm


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


def save_conversation(conv, user_email=None) -> bool:
    if not conv.user_turns:
        return False
    user_email = user_email or get_user_email()
    if not user_email:
        return False
    date = datetime.today().strftime('%m/%d/%Y')
    # TODO: maybe send as attachment instead?
    quickmail(f'Your conversation with {conv.current_persona} ({date}).',
              message=conv.full_conversation(),
              to_email=user_email,
              from_email=EMAIL)
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
            deco = ask.intent(name, mapping=mapping, **ask_kwargs)
            return deco(func)(*args, **kwargs)
        return wrapper
    return decorator


# TODO rm
@app.route('/')
def home():
    # TODO: docs
    app.logger.warning('>>> IN HOME')
    print('IN HOME')
    return 'home'


@app.route('/health')
def health():
    # TODO: docs
    print('IN HEALTH')
    return 'Jabberwocky is running.'


@ask.launch
def launch():
    # TODO: docs
    app.logger.warning('>>> IN LAUNCH')
    session.attributes['kwargs'] = conv._kwargs
    # return question('Who would you like to speak to?')
    # TODO
    tmp = question('Who would you like to speak to?')
    print(tmp)
    return tmp


@intent('choosePerson')
def choose_person(person):
    # TODO: docs
    if person not in conv:
        # TODO: new endpoint needed to handle answer to this case?
        return question(f'I don\'t see anyone named {person} in your '
                        f'contacts. Would you like to create a new contact?')

    conv.start_conversation(person)
    return question(f'I\'ve connected you with {person}.')


@intent('chooseModel')
def choose_model(model):
    # TODO: docs
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
    # TODO: docs
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


@ask.session_ended
def end_session():
    return '{}', 200


def exit():
    saved = False
    if session.attributes.get('should_end') and \
                session.attributes.get('should_save'):
            # TODO: have user configure this at some point earlier?
            saved = save_conversation(conv, session.attributes['email'])
    return saved


if __name__ == '__main__':
    app.logger.warning(f'>>> MAIN: {conv.personas()}') # TODO: rm
    app.run(debug=True)