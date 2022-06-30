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
