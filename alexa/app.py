"""
WIP: trying Flask-Ask since I'm not sure how to run the lambda_function locally
of if that's even possible. App relies on local filesystem a lot at the moment
so it might be more convenient to run locally with ngrok anyway.
"""

from collections import deque
from datetime import datetime
from functools import wraps, partial
import logging
from pathlib import Path

from flask import Flask, request
from flask_ask import Ask, question, context, statement
import requests

from config import EMAIL, LOG_FILE, HF_API_KEY
from htools import params, quickmail, save, MultiLogger, Callback, callbacks, \
    tolist, listlike, func_name, decorate_functions, debug as debug_decorator
from jabberwocky.openai_utils import ConversationManager, query_gpt_j,\
    query_gpt_neo, PromptManager
from utils import slot, Settings, model_type


class IntentCallback(Callback):
    # TODO: docs
    # TODO: maybe move to utils? But relies on state var. Could pass that to
    # init maybe?

    def __init__(self, ask):
        self.ask = ask

    def setup(self, func):
        pass

    def on_begin(self, func, inputs, output=None):
        self.ask.logger.info('\n' + '-' * 79)
        self.ask.logger.info('ON BEGIN')
        self.ask.func_dedupe(func)
        self._print_state(func)

    def on_end(self, func, inputs, output=None):
        state.prev_intent = self.ask.intent_name(func)
        self.ask.logger.info('\nON END')
        self._print_state()

    def _print_state(self, func=None):
        if func:
            self.ask.logger.info(f'Cur intent: {self.ask.intent_name(func)}')
        self.ask.logger.info(f'Prev intent: {state.prev_intent}')
        self.ask.logger.info(f'State: {state}')
        self.ask.logger.info(f'Queue: {self.ask._queue}\n')


class CustomAsk(Ask):
    """Slightly customized version of flask-ask's Ask object. See `intent`
    method for a summary of main changes.

    # TODO: move to utils? Depends on if we can move IntentCallback (see its
    docstring).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Unlike flask app.logger, this writes to both stdout and a log file.
        # We ensure it's empty at the start of each session.
        Path(LOG_FILE).unlink()
        self.logger = MultiLogger(LOG_FILE, fmode='a')
        # Decorator that we use on each intent endpoint.
        self._callbacks = callbacks([IntentCallback(self)])
        self._func2intent = {}
        self._intent2funcname = {}
        # Deque of functions probably can't/shouldn't be sent back and forth
        # in http responses, which I think `session` is, so we store this here
        # instead of in settings object. We must clear it with func_clear()
        # method when launching the skill because previously pushed functions
        # can persist otherwise.
        self._queue = deque()

    def func_push(self, *funcs):
        """Schedule a function (usually NOT an intent - that should be
        recognized automatically) to call after a user response. We push
        functions so that delegate(), yes() and no() know where to direct the
        flow to.

        A function can only occur in the queue once at any given time.
        Duplicates will not be added - we simply log a warning and continue.
        This is to handle situations like the following:
        1. Skill launch pushes choose_person into the queue.
        2. User ignores this and changes a setting instead, thereby attempting
        to push another choose_person call into the queue.

        Parameters
        ----------
        funcs: FunctionType(s)
            These should usually not be intents because those should already be
            recognized by Alexa. Pushing non-intents into the queue is useful
            if we want to say something to prompt the user to provide a value
            (guessing this is related to what elicit_slot in
            flask-ask does, but I couldn't figure out that interface).

            Pushing intents into the queue is only useful as a fallback - if
            the user utterance mistakenly is not matched with any intent and
            falls through to delegate(), it should then be forwarded to the
            correct intent.
        """
        for func in funcs:
            if func in self._queue:
                self.logger.warning(f'Tried to add function {func} to the '
                                    f'queue when it is alread present.')
            else:
                self._queue.extend(funcs)

    def func_pop(self):
        """

        Returns
        -------

        """
        try:
            return self._queue.popleft()
        except IndexError:
            self.logger.warning('Tried to pop chained function from empty '
                                'queue.')

    def func_clear(self):
        """Call this at the end of a chain of intents."""
        self._queue.clear()
        # We set this to an empty dict in launch so we shouldn't need to check
        # if it's None.
        state.kwargs.clear()

    def func_dedupe(self, func):
        """If we enqueue an intent and Alexa recognizes it by itself
        (without the help of delegate()),the intent function remains in the
        queue and would be called the next time we hit delegate()
        (not what we want). This method is auto-called before each intent is
        executed so we don't call it twice in a row by accident.

        Warning: this means you should NEVER have a function push itself into
        the queue.
        """
        # Slightly hacky by this way if one or both functions is decorated,
        # we should still be able to identify duplicates.
        if self._queue and func_name(self._queue[0]) == func_name(func):
            self.func_pop()

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


# TODO: maybe change back to debug eventually? Trying to unclutter terminal
# output bc it's making it harder to debug.
logging.getLogger('flask_ask').setLevel(logging.WARNING)
app = Flask(__name__)
# Necessary to make session accessible outside endpoint functions.
app.app_context().push()
ask = CustomAsk(app, '/')
conv = ConversationManager(['Albert Einstein']) # TODO: load all personas?
gpt = PromptManager(['punctuate_alexa'], verbose=False)
state = Settings()


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


@ask.launch
def launch():
    """Runs when user starts skill with command like 'Alexa, start Voice Chat'.
    """
    state.set('global', **conv._kwargs)
    # TODO: might want to change this eventually, but for now use free model
    # by default.
    conv.end_conversation()
    state.set('global', mock_func=query_gpt_j)
    state.kwargs = {}
    state.auto_punct = True
    # Make sure to clear queue after setting state.kwargs.
    ask.func_clear()
    # state.email = get_user_info() # TODO: revert from hardcoded to real
    for k, v in get_user_info().items():
        print('k,v =', k, v)
        setattr(state, k, v)
    # state.email = 'hmamin55@gmail.com'
    print('LAUNCH, email=', state.email) # TODO rm
    question_txt = _choose_person_text()
    return question(f'Welcome to Quick Chat. {question_txt}')\
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
    # a reply.
    if conv.current_persona:
        # Assume this is a regular reply in the midst of a conversation that
        # just happens to consist of only a name.
        return _reply(prompt=person)

    if person not in conv:
        state.kwargs = {'person': person}
        ask.func_push(_generate_person)
        return question(f'I don\'t see anyone named {person} in your '
                        f'contacts. Would you like to create a new contact?')

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


@ask.intent('enableAutoPunctuation')
def enable_punctuation():
    state.auto_punct = True
    return _maybe_choose_person('I\'ve enabled automatic punctuation.',
                                'Now, who would you like to speak to?')


@ask.intent('disableAutoPunctuation')
def disable_punctuation():
    state.auto_punct = False
    return _maybe_choose_person('I\'ve disabled automatic punctuation.',
                                'Now, who would you like to speak to?')


def _maybe_choose_person(msg='', choose_msg=''):
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
    return _maybe_choose_person(
        choose_msg='Now, who would you like to speak to?'
    )


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
    return _maybe_choose_person(msg, 'Now, who would you like to speak to?')


@ask.intent('endChat')
def end_chat():
    """End conversation with the current person.

    Sample utterance:
    "Lou, end chat."
    "Lou, hang up."
    """
    ask.func_push(_end_chat)
    return question('Would you like me to send you a transcript of your '
                    'conversation?')


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
    decorate_functions(debug_decorator)
    app.run(debug=True)
