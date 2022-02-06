"""
WIP: trying Flask-Ask since I'm not sure how to run the lambda_function locally
of if that's even possible. App relies on local filesystem a lot at the moment
so it might be more convenient to run locally with ngrok anyway.
"""

from functools import wraps
import logging

from flask import Flask
from flask_ask import Ask, statement, question, session, context, request

from htools import params
from jabberwocky.openai_utils import ConversationManager


logging.getLogger().setLevel(logging.INFO)
app = Flask(__name__)
ask = Ask(app, '/')
conv = ConversationManager(['Albert Einstein']) # TODO: load all personas?
app.logger.warning('>>> Loading globals') # TODO: rm


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
    app.logger.warning('>>> IN HOME')
    print('IN HOME')
    return 'home'


@app.route('/health')
def health():
    print('IN HEALTH')
    return 'Jabberwocky is running.'


@ask.launch
def launch():
    app.logger.warning('>>> IN LAUNCH')
    session.attributes['kwargs'] = conv._kwargs
    # return question('Who would you like to speak to?')
    # TODO
    tmp = question('Who would you like to speak to?')
    print(tmp)
    return tmp


@intent('choosePerson')
def choose_person(person):
    return statement(f'Choose person {person}.')

    if person not in conv:
        return question(f'I don\'t see anyone named {person} in your '
                        f'contacts. Would you like to create a new contact?')

    conv.start_conversation(person)
    return statement(f'I\'ve connected you with {person}.')


@intent('chooseModel')
def choose_model(model):
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
    # TODO: error handling
    session.attributes['kwargs']['max_tokens'] = int(length)
    return statement(f'Choose length {length}.')


@intent('changeTemperature')
def change_temperature(temperature):
    error_msg = 'Please choose a number greater than zero and ' \
                'less than or equal to one.'
    try:
        temperature = float(temperature)
        assert 0 < temperature <= 1
    except (TypeError, AssertionError) as e:
        if isinstance(e, TypeError):
            error_msg = ('I didn\'t recognize that temperature value. ' +
                         error_msg)
        return statement(error_msg)

    session.attributes['kwargs']['temperature'] = temperature
    return statement(f'I\'ve adjusted your temperature to {temperature}.')


@ask.session_ended
def end_session():
    return '{}', 200


if __name__ == '__main__':
    app.logger.warning(f'>>> MAIN: {conv.personas()}') # TODO: rm
    app.run(debug=True)