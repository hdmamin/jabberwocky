"""Helpers used in our alexa app. Particularly important
- CustomAsk class: subclass of one of the core flask-ask classes. Lots of
functionality around callbacks/logging/etc. take place here.
- Settings class: used to maintain gpt settings throughout a session. There are
global-level, conversation-level, and person-level settings (recency is
prioritized when resolving the final gpt query kwargs). You can read more about
this system in its docstrings.
- build_utterance_map function: any time we update the dialogue model in the
alexa UI, we must download the model JSON, save it to
jabberwocky/data/alexa/dialogue_model.json, and pass it to this function. This
will save a new dict-like object that allows us to infer intents and slots when
user utterances are not recognized as invoking an existing intent.
"""

from collections import Mapping, deque, OrderedDict, ChainMap
from enum import Enum
from flask_ask import session, Ask, question
from functools import wraps, partial
from fuzzywuzzy import fuzz, process
from itertools import product
import logging
import pandas as pd
from pathlib import Path
import spacy
import sys
from werkzeug.local import LocalProxy

from htools.core import xor_none
from htools.meta import Callback, callbacks, params, MultiLogger, func_name,\
    deprecated, select, save
from htools.structures import FuzzyKeyDict, LSHDict
from jabberwocky.openai_utils import query_gpt3


POLLY_NAMES = {
    'F': {
        # Standard only.
        'Australian': 'Nicole',
        # Standard or Neural. Amy is also a decent option.
        'British': 'Emma',
        # Standard or Neural.
        'American': 'Salli'
    },
    'M': {
        # Standard only.
        'Australian': 'Russell',
        # Standard or Neural.
        'British': 'Brian',
        # Standard or Neural.
        'American': 'Matthew'
    }
}

WORD2INT = FuzzyKeyDict(
    {num: i for i, num in enumerate([
        'one',
        'two',
        'three',
        'four',
        'five',
        'six',
        'seven',
        'eight',
        'nine',
        'ten',
        'eleven',
        'twelve',
        'thirteen',
        'fourteen',
        'fifteen',
        'sixteen',
        'seventeen',
        'eighteen',
        'nineteen',
        'twenty',
        'twenty one',
        'twenty two',
        'twenty three',
        'twenty four',
        'twenty five',
        'twenty six',
        'twenty seven',
        'twenty eight',
        'twenty nine',
        'thirty',
        'thirty one',
        'thirty two',
        'thirty three',
        'thirty four',
        'thirty five',
        'thirty six',
        'thirty seven',
        'thirty eight',
        'thirty nine',
        'forty',
        'forty one',
        'forty two',
        'forty three',
        'forty four',
        'forty five',
        'forty six',
        'forty seven',
        'forty eight',
        'forty nine',
        'fifty',
        'fifty one',
        'fifty two',
        'fifty three',
        'fifty four',
        'fifty five',
        'fifty six',
        'fifty seven',
        'fifty eight',
        'fifty nine',
        'sixty',
        'sixty one',
        'sixty two',
        'sixty three',
        'sixty four',
        'sixty five',
        'sixty six',
        'sixty seven',
        'sixty eight',
        'sixty nine',
        'seventy',
        'seventy one',
        'seventy two',
        'seventy three',
        'seventy four',
        'seventy five',
        'seventy six',
        'seventy seven',
        'seventy eight',
        'seventy nine',
        'eighty',
        'eighty one',
        'eighty two',
        'eighty three',
        'eighty four',
        'eighty five',
        'eighty six',
        'eighty seven',
        'eighty eight',
        'eighty nine',
        'ninety',
        'ninety one',
        'ninety two',
        'ninety three',
        'ninety four',
        'ninety five',
        'ninety six',
        'ninety seven',
        'ninety eight',
        'ninety nine',
        'one hundred'
    ], start=1)}
)


NLP = spacy.load('en_core_web_sm', disable=('parser', 'tagger'))


class custom_question(question):
    """Just like flask_ask.question except it handles emotion tags. The default
    version returns a json response marking the outputSpeech type as
    'plaintext' (or something like that) but we need it to be 'SSML'.

    """

    def __init__(self, speech, is_ssml=False):
        super().__init__(speech)
        self._set_ssml(is_ssml)

    def reprompt(self, reprompt, is_ssml=False):
        super().reprompt(reprompt)
        self._set_ssml(is_ssml, 'reprompt')
        return self

    def _set_ssml(self, is_ssml=False, attr=''):
        """Update response dict to use ssml if appropriate. Flask-ask
        doesn't natively recognize amazon's voice markup language so we do this
        here instead.

        Parameters
        ----------
        is_ssml: bool
            If False, no changes are needed. If True, we adjust the response
            dict.
        attr: str
            If truthy, we find this key in the response dict and edit that
            rather than the base level of the dict. Only known use case atm
            is passing in 'reprompt' in the reprompt method.
        """
        if not is_ssml:
            return
        if attr:
            data = self._response[attr]
        else:
            data = self._response
        data = data['outputSpeech']
        if data['type'] != 'SSML':
            text = data.pop('text')
            data['type'] = 'SSML'
            data['ssml'] = text


def select_polly_voice(current_meta, threshold=.8,
                       other={'F': 'Australian', 'M': 'British'}):
    """Select name of polly voice to use based on a person's gender and
    nationality.

    Setting male default to British for now because the male Australian voice
    turned out to be much more synthetic sounding than I initially realized.
    Maybe change it back in the future if Amazon improves it. Female Australian
    voice needs more qualitative evaluation from me.
    Also, `other` is technically mutable but it doesn't matter since we don't
    alter it in any way inside the function.

    Parameters
    ----------
    current_meta: dict
        ConversationManager.current_meta attribute which stores information
        about the person currently being spoken to including "gender" and
        "nationality".
    threshold: float
        Value in (0.0, 1.0) that defines the minimum score from our fuzzy
        matcher when finding a matching voice country. If the highest
        similarity is below this threshold, we default to `other`.
    other: str
        One of the countries in POLLY_NAMES[gender].keys(). If a close match
        isn't found, we default to this. (Basically, Polly has limited
        countries so as of 5/24/22, all non-US/British people get "Austrlian"
        which sort of serves as a catch-all "funky/other" category.
    """
    country2voice = POLLY_NAMES[current_meta['gender']]

    # Wikipedia usually calls people these people English. Need to map to Polly
    # name.
    country = current_meta['nationality'].replace('English', 'British')
    score_pairs = [(match, score / 100) for match, score
                   in process.extract(country, list(country2voice))]
    match, score = score_pairs[0]
    if score < threshold:
        if isinstance(other, Mapping):
            match = other[current_meta['gender']]
        elif isinstance(other, str):
            match = other
        else:
            raise TypeError('`other` should have type str or dict. You '
                            f'passed in {other}.')
    if match not in country2voice:
        raise KeyError(f'Default polly country {other} is not '
                       f'available. Pick one of {country2voice}.')
    return country2voice[match]


def emotion(text, pred=None, emo_pipe=None, logger=None):
    xor_none(pred, emo_pipe)
    pred = pred or emo_pipe(text)
    if isinstance(pred, (tuple, list)):
        pred = pred[0]
    if logger:
        logger.info(f'\nEMOTION PREDICTION: {pred}')
    res = {'emotion': None,
           'intensity': None}
    labels = ['sadness', 'disgust', 'fear', 'joy']
    if pred['score'] < .65 or pred['label'] not in labels:
        return res
    if pred['label'] == 'joy':
        if '!' in text:
            res['emotion'] = 'excited'
            res['intensity'] = 'high'
        elif pred['score'] >= .9:
            res['emotion'] = 'excited'
            res['intensity'] = 'low'
    else:
        res['emotion'] = 'disappointed'
        if pred['label'] == 'sadness':
            res['intensity'] = 'high' if pred['score'] >= .75 else 'low'
        else:
            res['intensity'] = 'high' if pred['score'] >= .9 else 'low'
    return res


def voice(text, current_meta, polly_name=None, select_voice=True,
          emo_pipe=None, **kwargs):
    """Add voice tags to use a custom Amazon Polly voice.

    Returns
    -------
    str: the input text formatted with amazon's
    voice markup language to use either a custom voice or one of a limited set
    of emotions (if appropriate).
    """
    open_tags = ['<speak>']
    close_tags = ['</speak>']
    # We can either use a custom voice or custom emotions. For now, I set
    # custom voice as the default - the emotional range is rather limited and
    # only affects a small percentage of responses.
    if select_voice:
        name = polly_name or select_polly_voice(current_meta, **kwargs)
        open_tags.append(f'<voice name="{name}">')
        close_tags.append('</voice>')
    else:
        logger = getglobal('ask.logger')
        tags = emotion(text, emo_pipe=emo_pipe, logger=logger)
        if tags['emotion']:
            open_tags.append(
                f'<amazon:emotion name="{tags["emotion"]}" '
                f'intensity="{tags["intensity"]}">'
            )
            close_tags.append('</amazon:emotion>')
        logger.info(f'TAGS: {tags}')
    return f"{''.join(open_tags)}{text}{''.join(close_tags[::-1])}"


def detokenize(tokens, punct=set('.,;:')):
    # NLTK has built in detokenizer but it was adding spaces before commas and
    # things like that.
    res = ''
    for tok in tokens:
        if tok not in punct:
            res += ' '
        res += tok
    return res.strip(' ')


"""
Note: these three get_ functions were part of an initial attempt at extracting
slots when alexa failed. This is the same method that caused the need for my
SlotType class. See commit on 3/21/22 with message "checkpoint: pre-delete
type hint-based slot extraction" for how this worked. Never fully finished 
this so don't expect everything to work even at that commit - notice the
inconsistent interfaces where some return a dict with 'value' and 
'disambiguation' keys and others don't.

Ultimately switched to a different approach where our fuzzy dict maps sample 
utterances to their slot values and we just take the closest utterance's slots
rather than trying to extract them each time.

6/19/22 update: we actually do use get_number() now, though the other slot
extraction functions remain unused (for now at least). Some fields like
max_length have a huge number of possible numeric values and the number of 
sample utterances was starting to explode, making our fuzzydict noticeably 
slow. LSHDict was faster but less accurate. See nb20 for other approaches - 
they might become necessary again if we add many new sample utterances, new
intents, or more valid values per slot.
"""


def get_number(text):
    """Extract a numeric value from an utterance. This is used by infer_intent
    when the inferred intent is "changeTemperature" or "changeMaxLength" to
    extract the numeric slot value.

    Returns
    -------
    dict: Key "value" points to an integer if found, or None if not. In the
    latter case, an additional "disambiguation" key is added, containing a
    (possibly empty) list candidates found. We expect there to be only 1 number
    found - finding multiple therefore triggers this second case
    (i.e. value=None).

    Examples
    --------
    lou set temperature to 95 -> {'value': 95}
    """
    nums = [t.text for t in NLP(text) if t.like_num]
    if len(nums) == 1:
        return {'value': int(nums[0])}
    return {'value': None,
            'disambiguation': nums}


@deprecated
def get_name(text, skip={'Lou', 'lou'}):
    names = [ent.text for ent in NLP(text).ents
             if ent.label_ == 'PERSON' and ent.text not in skip]
    if len(names) == 1:
        return {'value': names[0]}
    return {'value': None,
            'disambiguation': names}


@deprecated
def get_backend(
        text, scorer=fuzz.ratio,
        n=3,
        thresh=80,
        backends=('goose ai', 'open ai', 'hobby', 'hugging face'),
        skip=('lou', 'backend', 'back end', 'change', 'switch', 'set', 'use')
):
    # Putting imports inside since this is deprecated anyway. Don't want to
    # slow down app.py when it imports this module.
    import re
    from nltk.tokenize import word_tokenize
    from htools import ngrams

    text = text.lower()
    for skip_word in skip:
        text = text.replace(skip_word, '')
    text = re.sub('  *', ' ', text)

    one_grams = word_tokenize(text)
    two_grams = [detokenize(pair) for pair in
                 ngrams(one_grams, n=2, drop_last=True)]
    res = {}
    for backend in backends:
        if ' ' in backend:
            res[backend] = process.extract(backend, two_grams, scorer=scorer,
                                           limit=n)
            backend = backend.replace(' ', '')
        res[backend] = process.extract(backend, one_grams, scorer=scorer,
                                       limit=n)
    b2matches = {k: [pair for pair in v if pair[1] >= thresh] or [v[0]]
                 for k, v in res.items()}
    return sorted(b2matches.items(), key=lambda x: x[1][0][-1], reverse=True)


@deprecated(msg='get_scope was never finished and is now deprecated. It\'s '
            'used in SlotType\'s definition so I don\'t delete it, though '
            'that class is no longer used. Just want to keep it as a '
            'reference since it does some pretty interesting things, '
            'code-wise.')
def get_scope(text):
    return {}


class SlotType(Enum):
    """Bit of a weird way to use enum and not really necessary, just trying out
    different ways to handle this kind of task. Gives us a nice user
    interface for specifying slot types, e.g. model=SlotType.NUMBER,
    while easily mapping these to functions without any extra indexing.
    """
    # Kind of hacky but regular functions get recognized as methods and lose
    # enum functionality. Partial basically just hides them.
    NAME = partial(get_name)
    NUMBER = partial(get_number)
    BACKEND = partial(get_backend)
    SCOPE = partial(get_scope)


@deprecated
def make_slot_func(func):
    """Given a function (an alexa intent/flask endpoint), uses the type
    annotations to cconstruct a function that extracts slot values from an
    utterance string. Used in ask.intent().
    """
    def slot_func(text):
        return {k: v.value(text) for k, v in func.__annotations__.items()
                if isinstance(v, SlotType)}
    return slot_func


class IntentCallback(Callback):
    """Add some extra functionality at start and end of each intent function.
    Initially tried to implement this by decorating ask.intent but that seems
    to use some python black magic that makes that approach very tricky.

    Specifically, this logs the current intent and the previous intent both
    before and after execution, and handles updating and deduplicating the
    previous intent attribute as well.
    """

    def __init__(self, ask, state):
        self.ask = ask
        self.state = state

    def setup(self, func):
        pass

    def on_begin(self, func, inputs, output=None):
        self.ask.func_dedupe(func)
        self._print_state(on_begin=True, func=func)
        # Update this after logging but before on_end. Sometimes an intent
        # function calls other intent functions within it and it was getting
        # confusing when prev_intent wasn't updated til afterwards.
        self.state.prev_intent = self.ask.intent_name(func)
        self.ask.stack_size += 1

    def on_end(self, func, inputs, output=None):
        self.ask.stack_size -= 1
        if self.ask.stack_size < 0:
            self.ask.logger.error('Ask.stack_size should always be >= 0.')
        self._print_state(on_begin=False, func=func)

    def _print_state(self, on_begin, func=None):
        func_name_ = self.ask.intent_name(func)
        if on_begin:
            self.ask.logger.info('-' * 79)
            start_msg = 'ON BEGIN'
        else:
            start_msg = f'\nON END'
        self.ask.logger.info(f'{start_msg} ({func_name_})')
        self.ask.logger.info(f'Prev intent: {self.state.prev_intent}')
        self.ask.logger.info(f'State: {dict(self.state)}')
        self.ask.logger.info(f'Queue: {self.ask._queue}\n\n')


class IndentedFormatter(logging.Formatter):
    """Custom formatter for ask.logger so that all logging is indented by the
    appropriate amount, depending on the size of the intent stack.
    """

    def __init__(self, ask, filler=' ', *args, **kwargs):
        self.ask = ask
        self.filler = filler
        super().__init__(*args, **kwargs)

    def format(self, record):
        msg = super().format(record)
        indent = max(0, self.ask.stack_size * 4) * self.filler
        return '\n'.join(indent + line for line in msg.splitlines())


class CustomAsk(Ask):
    """Slightly customized version of flask-ask's Ask object. See `intent`
    method for a summary of main changes.
    """

    def __init__(self, app, route, state, log_file, filler=' ',
                 *args, **kwargs):
        super().__init__(app=app, route=route, *args, **kwargs)
        # Unlike flask app.logger, this writes to both stdout and a log file.
        # We ensure it's empty at the start of each session.
        self.logger = self.get_logger(log_file, filler=filler)
        # Decorator that we use on each intent endpoint.
        self.state = state
        self._callbacks = callbacks([IntentCallback(self, state=state)])
        # str -> str
        self._func2intent = {}
        self._intent2funcname = {}
        # Deque of functions probably can't/shouldn't be sent back and forth
        # in http responses, which I think `session` is, so we store this here
        # instead of in settings object. We must clear it with func_clear()
        # method when launching the skill because previously pushed functions
        # can persist otherwise.
        self._queue = deque()
        self.stack_size = 0

    def get_logger(self, log_file, filler):
        # Indent log statements depending on our depth in the intent stack.
        # File won't exist the first time we run this on a new machine.
        try:
            Path(log_file).unlink()
        except FileNotFoundError:
            pass
        logger = MultiLogger(log_file, fmode='a')
        formatter = IndentedFormatter(self, filler=filler)
        for handler in logger.handlers:
            handler.setFormatter(formatter)
        return logger

    def intent2func(self, intent_name:str):
        """Map from name of an intent to the function (flask endpoint) that
        executes it. This is used when inferring intents.

        Returns
        -------
        FunctionType
        """
        return getattr(sys.modules['__main__'],
                       self._intent2funcname[intent_name])

    def func_push(self, func, **kwargs):
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
        func: FunctionType
            This should usually not be an intent because those should already
            be recognized by Alexa. Pushing non-intents into the queue is
            useful if we want to say something to prompt the user to provide a
            value (guessing this is related to what elicit_slot in
            flask-ask does, but I couldn't figure out that interface).

            Pushing intents into the queue is only useful as a fallback - if
            the user utterance mistakenly is not matched with any intent and
            falls through to delegate(), it should then be forwarded to the
            correct intent.
        """
        if func in (func_ for func_, kwargs_ in self._queue):
            self.logger.warning(f'Tried to add function {func} to the '
                                f'queue when it is already present. '
                                f'Skipping push operations.')
        else:
            self._queue.append((func, kwargs))

    def func_pop(self):
        """Remove the first function in the queue (the oldest one, and
        therefore the next to be processed.

        Returns
        -------
        FunctionType, dict: second item are the kwargs to pass to the first
        item.
        """
        try:
            return self._queue.popleft()
        except IndexError:
            self.logger.warning('Tried to pop chained function from empty '
                                'queue.')
            # Return 2 items since we try to assign result to `func, kwargs`.
            return None, {}

    def func_clear(self):
        """Call this at the end of a chain of intents."""
        self._queue.clear()

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
        if self._queue and func_name(self._queue[0][0]) == func_name(func):
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
        - Automatically map slot names from title case to lowercase. AWS
        console seems to enforce some level of capitalization that I'd prefer
        not to use in all my python code. (UPDATE: this doesn't seem to be true
        now, not sure why I thought it was? Could get rid of this but I guess
        it's fine.)
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


class Settings(Mapping):
    """Object for tracking settings throughout an alexa session. This serves
    2 purposes:

    1. Cleanly resolve 3 different levels of query settings into 1 set of
    arguments to call GPT3 with. User can make changes at the global level
    (persist until they quit the skill), person level (persist until they
    switch to talk to a different person during the same skill session),
    or conversation level (persist until the current conversation ends).
    2. Provide a slightly cleaner interface to session.attributes. Concretely,
    the first version is much less unwieldy than the second, IMO:

    ```
    if state.prev_intent == 'quit' and state['should_email']:
    ```

    ```
    if session.attributes['prev_intent'] == 'quit' \
            and session.attributes['should_email']:
    ```

    Examples
    --------
    Create a new object (favor a short name due to frequency of use).
    >>> state = Settings()

    Set model kwargs:
    >>> state.set(scope, max_tokens=6)
    Get model kwargs:
    >>> state['max_tokens']

    Set reserved attributes:
    >>> state.prev_intent = 'choose_person'
    Get reserved attributes:
    >>> state.prev_intent

    Unpack model kwargs:
    >>> res = query_gpt3(prompt, **state)
    """

    def __init__(
            self, global_=None, person_=None, conversation_=None,
            reserved_keys=('prev_intent', 'email', 'auto_punct', 'polly_voice')
    ):
        """

        Parameters
        ----------
        global_: None or dict
            Global level settings.
        person_: None or dict
            Person level settings.
        conversation_: None or dict
            Conversation level settings.
        reserved_keys: Iterable[str]
            Attribute names that will use dot notation for get/set calls.
            See class-level docstring.
        """
        # Alter underlying dict directly to avoid recursion errors. Custom
        # __setattr__ method relies on custom __getattr__ so we run into
        # problems otherwise.
        self.__dict__['reserved_keys'] = set(reserved_keys)
        self._global = dict(global_ or {})
        self._person = dict(person_ or {})
        self._conversation = dict(conversation_ or {})

        # This is populated in _resolve_state().
        self.state = ChainMap()
        # The start (as in state_queue.move_to_end(scope, last=False)) is the
        # highest priority position.
        self.state_queue = OrderedDict({
            'conversation': self._conversation,
            'person': self._person,
            'global': self._global,
        })
        self._resolve_state()

    def __setattr__(self, key, value):
        if key in self.reserved_keys:
            session.attributes[key] = value
        else:
            super().__setattr__(key, value)

    def __getattr__(self, key):
        # Made a slightly unusual choice: reserved keys always just return
        # None if user tries to access them before they're set. We can't
        # explicitly initialize them to None (which would probably be ideal)
        # because session.attributes is initially set to None and flask seems
        # to overwrite it at some point while running the app.
        if key in self.reserved_keys:
            return session.attributes.get(key, None)
        raise AttributeError

    def init_settings(self, conv=None,
                      args=('model', 'temperature', 'max_tokens',
                            'frequency_penalty'), **kwargs):
        """Don't call this in __init__ automatically because flask session
        object is not yet not available. Instead, we call it in the
        reset_app_state function in app.py.
        """
        if conv:
            new_kwargs = select(conv._kwargs, keep=args)
        else:
            new_kwargs = getdefaults(query_gpt3, *args)
        kwargs.update(new_kwargs)
        self.set('global', **kwargs)

    def on_conv_start(self, conv):
        """Call AFTER calling conv_manager.start_conversation(name).

        Parameters
        ----------
        conv: jabberwocky.openai_utils.ConversationManager
        """
        # Clear conversation kwargs here too just to be safe (we also do this
        # in on_conv_end). Person-level kwargs now take priority.
        self.clear('conversation')
        self._person = conv.name2kwargs[conv.current['persona']]
        self.state_queue.move_to_end('person', last=False)
        self._resolve_state()

    def on_conv_end(self):
        # Global settings take priority now. Clearing conversation kwargs
        # removes the need to move it in the queue explicitly.
        self.clear('conversation')
        # Remove reference to the dict we obtained from conv.name2kwargs.
        # State queue gets updated by _resolve_state() call.
        self._person = {}
        self.state_queue.move_to_end('global', last=False)
        self._resolve_state()

    @classmethod
    def clone(cls, settings):
        return cls(settings._global.copy(),
                   settings._person.copy(),
                   settings._conversation.copy(),
                   settings.reserved_keys.copy())

    def _resolve_state(self):
        # The top of the stack takes priority. Resolution order is more
        # about change *recency* than some inherent hierarchy of one state
        # taking precedence over another. We must refresh values in state queue
        # in case _person dict has changed.
        self.state_queue = OrderedDict({key: getattr(self, f'_{key}')
                                        for key in self.state_queue})
        self.state = ChainMap(*self.state_queue.values())

    def __getitem__(self, key):
        return self.state[key]

    def __iter__(self):
        return iter(self.state)

    def __len__(self):
        return len(self.state)

    def get(self, key, default=None):
        return self.state.get(key, default)

    def __setitem__(self, key, val):
        raise RuntimeError('You cannot use __setitem__ on a Settings object. '
                           'Instead, use the `set` method.')

    def __delitem__(self, key):
        raise RuntimeError('You cannot use __delitem__ on a Settings object. '
                           'Instead, use the `delete` method.')

    def _set(self, scope, key, val, lazy=False):
        self.state_queue[scope][key] = val
        if not lazy:
            self._resolve_state()

    def set(self, scope, **kwargs):
        # Set multiple kwargs but only resolve once. Slightly more efficient
        # than resolving once for every kwarg.
        for k, v in kwargs.items():
            self._set(scope, k, v, lazy=True)
        # Most recent changes always take priority.
        self.state_queue.move_to_end(scope, last=False)
        self._resolve_state()

    def pop(self, scope, key, default=None):
        # 6/12/22: forget what this was originally intended for but it doesn't
        # look like it's currently in use. Keep for now just in case.
        res = self.state_queue[scope].pop(key, default)
        self.state_queue.move_to_end(scope, last=False)
        self._resolve_state()
        return res

    def clear(self, *states):
        """Reset settings.

        Parameters
        ----------
        states: str
            Optionally pass in one or more states
            ('global', 'conversation', and/or 'person') to clear. Otherwise
            we clear all states.
        """
        states = set(states or self.state_queue.keys())
        for name, state in self.state_queue.items():
            if name in states:
                state.clear()
        self._resolve_state()

    def __repr__(self):
        return f'Settings({self.state})'


def getglobal(attr):
    """Basically getattr where the object is the __main__ module and we can
    specify nested attributes (e.g. getglobal('ask.logger') works). This is
    intended for globals defined in the file that imports this function, which
    would not be available if we used globals().
    """
    parts = attr.split('.')
    obj = sys.modules['__main__']
    for part in parts:
        obj = getattr(obj, part)
    return obj


def slot(request, name, lower=True, default=''):
    """Extract a slot value from flask_ask response object. I thought the
    library was supposed to do this automatically but it doesn't seem to.

    Parameters
    ----------
    request: werkzeug.local.LocalProxy or dict
    name: str
        Name of slot to extract. This was set in the Alexa UI.
    lower: bool
        If True, lowercase the resulting value before returning.
    default: str
        Value to return if slot could not successfully be extracted.
    """
    if isinstance(request, LocalProxy): request = request.get_json()
    logger = getglobal('ask.logger')
    failed_parse_symbol = '?'

    # If a function with slots is pushed into the queue and the user responds
    # in such a way that it should not actually be called, then this raises an
    # error. I do the minimum here to allow the rest of my error handling code
    # to take effect in the next try/except block. Without this, the function
    # would fail to return our default value in this (quite rare) situation.
    # The only time I encountered it was when purposefully responding with
    # something unintelligible to the question "Who would you like to speak
    # to?". Alexa mistakenly parsed it as "no" and proceeded to call
    # choose_person as planned, which calls slot('person') and caused an error.
    try:
        slots_ = request['request']['intent']['slots']
    except KeyError:
        slots_ = {}

    try:
        # I think AMAZON.Number slots don't have 'resolutions' key. Also,
        # starting to think maybe 'value' is more reliable anyway? Observed one
        # instance where 'value' was the right match but first resolution was
        # wrong.
        logger.info(f'SLOTS\nname={name}\nslots_={slots_}\n')
        res = slots_[name]['value']
    except (KeyError, IndexError):
        try:
            resolved = slots_[name]['resolutions']['resolutionsPerAuthority']
            res = resolved[0]['values'][0]['value']['name']
        except Exception as e:
            logger.error(f'Slot parsing failed due to {e}.')
            res = failed_parse_symbol

    if lower: res = res.lower()
    logger.info(f'SLOT RESOLVED: {res}')
    return default if res == failed_parse_symbol else res


def model_type(state):
    """Identify ModelType from session state. Mostly used when communicating
    failed model change to user.

    Parameters
    ----------
    state

    Returns
    -------
    str or int: 0, 1, 2, 3, 'neo', or 'j'.
    """
    mock_func = state.get('mock_func', None)
    if mock_func:
        return mock_func.__name__.split('_')[-1]
    return state['model_i']


def build_utterance_map(model_json, fuzzy=True,
                        exclude_types=('AMAZON.Person', 'AMAZON.SearchQuery'),
                        save_=False, model_path='data/alexa/dialog_model.json',
                        meta_path='data/alexa/utterance2meta.pkl',
                        min_num=0, max_num=5):
    """Given a dictionary copied from Alexa's JSON Editor, return a
    dict or FuzzyKeyDict mapping each possible sample utterance to its
    corresponding intent. This allows our delegate() function to do some
    utterance validation before blindly forwarding an utterance to _reply() or
    the next queued function.

    Warning: because each intent may have several utterances and
    each utterance may contain multiple slots and each slot may have multiple
    sample values, the dimensionality can blow up quickly here.

    Parameters
    ----------
    model_json
    exclude_types: Iterable[str]
        One or more slot types where we want to exclude intents that contain
        any of them from the output map. For example, AMAZON.SearchQuery is
        meant to capture whole utterances matching no particular format as a
        fallback intent, so it wouldn't make sense to try to fuzzy match
        these utterances to an intent. I could see AMAZON.Person being included
        in some contexts but in this skill, we only use it for the choosePerson
        utterance which consists solely of a name. There really shouldn't be a
        reason to fuzzy match that.
    min_num: int
        For slots that have type AMAZON.NUMBER, we have to specify possible
        values in this function since they're not provided as sample values
        in the intent model. Min_num is the lowest possible value. All values
        between min_num and max_num (inclusive) will be included as possible
        values for numeric slots. Note that the dimensionality blows up
        *extremely* quickly. Luckily, fuzzy dict is fast enough that
        even searching thousands of utterances takes a negligible amount of
        time. If we really need to speed things up in the future (if we add
        many new intents, for example), we could switch to LSHDict and/or
        randomly sample a subset of numbers in range(min_num, max_num + 1). The
        latter is a bit risky though, since the nearest string match might at
        times depend more on phrasing than on number (i.e. if we say
        "Lou, change global temperature to 35" and we don't have that as a
        sample utterance, it's possible we end up ranking something like
        "Lou, change global temperature to 3" as a better match than
        "Lou, use global temperature 35".
    max_num: int
        See min_num explanation.

    Returns
    -------
    Dict: Maps sample utterance to dict containing 'intent' str and 'slots'
    dict.
    """
    exclude_types = set(exclude_types)
    model = model_json['interactionModel']['languageModel']
    type2vals = {type_['name']: [row['name']['value']
                                 for row in type_['values']]
                 for type_ in model['types']}
    type2vals['AMAZON.NUMBER'] = list(map(str, range(min_num, max_num + 1)))
    utt2meta = {}
    for intent in model['intents']:
        slot2vals = {}
        try:
            for slot_ in intent.get('slots', []):
                assert slot_['type'] not in exclude_types
                slot2vals[slot_['name']] = type2vals[slot_['type']]
        except AssertionError:
            continue

        # Replace all slot names with common slot values.
        for row in intent['samples']:
            for args in product(*slot2vals.values()):
                kwargs = dict(zip(slot2vals, args))
                utt2meta[row.format(**kwargs)] = {'intent': intent['name'],
                                                  'slots': kwargs}
    meta = FuzzyKeyDict(utt2meta) if fuzzy else utt2meta
    if save_:
        save(model_json, model_path)
        save(meta, meta_path)
    return meta


def infer_intent(
    utt,
    fuzzy_dict,
    n_keys=5,
    n_candidates=None,
    top_1_thresh=0.9,
    weighted_thresh=0.7,
):
    """
    Try to infer the user's intent from an utterance. Alexa should detect
    this automatically but it sometimes messes up. This also helps if the user
    gets the utterance slightly wrong, e.g. "Lou, set backend to goose ai"
    rather than "Lou, switch backend to goose ai".

    Parameters
    ----------
    utt
    fuzzy_dict
    n_keys
    top_1_thresh
    weighted_thresh

    Returns
    -------
    dict: Contains keys "intent", "confidence", "reason", and "res".
    Intent is the name of the closest matching intent if one was sufficiently
    close (empty string otherwise), confidence is a float between 0 and 1
    indicating our confidence in this being correct (sort of, not anything
    rigorous though; -1 if no matching intent is found), and reason is a string
    indicating our method for determining this ('top_1' means we found 1 sample
    utterance that was very close to the input, 'weighted' means that most of
    the nearest matching utterances tended to belong to the same intent, empty
    string means no matching intent was found). Res is always just the raw
    results of our fuzzy_dict similar() method call, a list of tuples
    containing all n_keys matching utterances, their corresponding intents,
    and similarity scores.
    """
    kwargs = dict(n_keys=n_keys, mode='keys_values_similarities')
    if isinstance(fuzzy_dict, LSHDict):
        kwargs['n_candidates'] = n_candidates
    res = fuzzy_dict.similar(utt, **kwargs)
    top_1_pct = res[0][-1] / 100
    res_final = {'res': res}
    if top_1_pct >= top_1_thresh:
        res_final.update(intent=res[0][1]['intent'],
                         slots=res[0][1]['slots'],
                         confidence=top_1_pct,
                         reason='top_1')
    else:
        df = pd.DataFrame(res, columns=['txt', 'intent', 'score'])
        df['slots'] = df.intent.apply(lambda x: x['slots'])
        df['intent'] = df.intent.apply(lambda x: x['intent'])
        weighted = df.groupby('intent').score.sum()\
            .to_frame()\
            .assign(pct=lambda x: x / (n_keys * 100))

    # Only consider weighted method if top 1 check was not satisfied.
    if 'intent' not in res_final:
        if weighted.pct.iloc[0] > weighted_thresh:
            intent = weighted.iloc[0].name
            slots = df.loc[df.intent == intent, 'slots'].iloc[0]
            res_final.update(intent=intent,
                             slots=slots,
                             confidence=weighted.iloc[0].pct,
                             reason='weighted')
        else:
            # In this case, confidence is a bit different but it's loosely
            # intended to mean "confidence that the utterance matched no
            # pre-defined intent". Value simply needs to be higher than
            # 1 - weighted_thresh.
            res_final.update(intent='',
                             slots={},
                             confidence=1 - weighted.iloc[0].pct,
                             reason='')
    if res_final['intent'] and 'Number' in res_final['slots']:
        res_final['slots']['Number'] = get_number(utt)['value']
        # Sometimes no number could be extracted. Unclear what confidence
        # score should be here so I just leave it unchanged.
        if res_final['slots']['Number'] is None:
            res_final['intent'] = ''
    return res_final


def getdefaults(func, *args):
    """Get default value for a function argument.

    Parameters
    ----------
    func: FunctionType
    arg: str

    Returns
    -------
    dict[str, any]: Maps parameter name(s) to default value(s).

    Examples
    --------
    >>> getdefaults(query_gpt3, 'temperature')
    {'temperature': 0.7}

    >>> getdefaults(query_gpt3, 'temperature', 'model')
    {'temperature': 0.7,
     'model': 0}
    """
    params_ = params(func)
    return {k: v.default for k, v in params_.items() if k in args}
