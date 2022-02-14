"""
Copied (with tweaks) from
https://developer.amazon.com/blogs/post/Tx14R0IYYGH3SKT/Flask-Ask-A-New-Python-Framework-for-Rapid-Alexa-Skills-Kit-Development
for debugging purposes. Just trying to run simplest app possible. In console,
I've named this skill "skill debugger".
"""

import logging
from random import randint

from flask import Flask, render_template
from flask_ask import Ask, statement, question, session


app = Flask(__name__)
ask = Ask(app, "/")
logging.getLogger("flask_ask").setLevel(logging.DEBUG)


@ask.launch
def new_game():
    welcome_msg = render_template('welcome', template_folder='alexa')
    return question(welcome_msg)


@ask.intent("YesIntent")
def next_round():
    numbers = [randint(0, 9) for _ in range(3)]
    round_msg = render_template('alexa/round', numbers=numbers)
    session.attributes['numbers'] = numbers[::-1]  # reverse
    return question(round_msg)


@ask.intent('HelloWorldIntent')
def hello():
    return question('Hello world!')


@ask.intent("AnswerIntent",
            convert={'first': int, 'second': int, 'third': int})
def answer(first, second, third):
    winning_numbers = session.attributes['numbers']
    if [first, second, third] == winning_numbers:
        msg = render_template('alexa/win')
    else:
        msg = render_template('alexa/lose')
    return statement(msg)


@ask.intent('AMAZON.FallbackIntent')
def fallback():
    app.logger.debug('>>> IN FALLBACK')
    return question('Could you repeat that?')


if __name__ == '__main__':

    app.run(debug=True)