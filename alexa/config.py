"""Constants used in our alexa skill. The email is used to send transcripts
to the user when desired. Note that in addition to the log file specified here,
GPT queries are also logged to files like `2022.06.25.jsonlines`. A new file is
generated for each day (the switch occurs at midnight) and each line in the
file corresponds to kwargs for a single gpt query.
"""

from htools import load

EMAIL = 'jabberwocky-alexa@outlook.com'
DEV_EMAIL = load('data/private/dev_email.txt').strip()
LOG_FILE = 'alexa/app.log'
# We randomly pick one of these to use in a reprompt whenever user takes too
# long to respond.
REPROMPTS = [
    'I know, it\'s a lot to take in.',
    'I can see you\'re thinking hard.',
    'I know you like to mull things over.',
    'I can see the gears turning.',
    'I\'m not going anywhere. Take your time.'
]
# Weird values/spellings here are mis-transcriptions I observed Alexa make.
NOBODY_UTTS = {'knobody', 'nobody', 'noone', 'no one', 'no1', 'no 1'}
