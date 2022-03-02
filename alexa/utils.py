from collections import Mapping
from flask_ask import session
from werkzeug.local import LocalProxy

from htools.structures import FuzzyKeyDict, DotDict


word2int = FuzzyKeyDict(
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
    if session.attributes['prev_intent' == 'quit' \
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

    def __init__(self, global_=None, person_=None, conversation_=None,
                 reserved_keys=('prev_intent', 'email', 'kwargs')):
        # Alter underlying dict directly to avoid recursion errors. Custom
        # __setattr__ method relies on custom __getattr__ so we run into
        # problems otherwise.
        self.__dict__['reserved_keys'] = set(reserved_keys)
        self._global = dict(global_ or {})
        self._person = dict(person_ or {})
        self._conversation = dict(conversation_ or {})

        # Cleaner interface than using getattr all the time.
        self._states = {
            'global': self._global,
            'person': self._person,
            'conversation': self._conversation
        }
        self.state = {}
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

    @classmethod
    def clone(cls, settings):
        return cls(settings._global.copy(),
                   settings._person.copy(),
                   settings._conversation.copy(),
                   settings.reserved_keys.copy())

    def _resolve_state(self):
        # Order matters here: we want global settings to take priority over
        # person-level settings, and both of those to take priority over
        # conversation level settings.
        self.state = {
            **self._states['conversation'],
            **self._states['person'],
            **self._states['global']
        }

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
        self._states[scope][key] = val
        if not lazy:
            self._resolve_state()

    def set(self, scope, **kwargs):
        # Set multiple kwargs but only resolve once. Slightly more efficient
        # than resolving once for every kwarg.
        for k, v in kwargs.items():
            self._set(scope, k, v, lazy=True)
        self._resolve_state()

    def pop(self, scope, key, default=None):
        res = self._states[scope].pop(key, default)
        self._resolve_state()
        return res

    def clear(self):
        for state in self._state.values():
            state.clear()
        self._resolve_state()

    def __repr__(self):
        return f'Settings({self.state})'


def slot(request, name, lower=True, default=''):
    """
    # TODO: docs

    Parameters
    ----------
    request: werkzeug.local.LocalProxy or dict
    name
    lower
    """
    if isinstance(request, LocalProxy): request = request.get_json()
    failed_parse_symbol = '?'
    slots_ = request['request']['intent']['slots']
    try:
        print('SLOTS\n', name, slots_, '\n')   # TODO: maybe rm
        resolved = slots_[name]['resolutions']['resolutionsPerAuthority']
        res = resolved[0]['values'][0]['value']['name']
    except (KeyError, IndexError):
        # I think AMAZON.Number slots don't have resolutions so we also check
        # this backup key.
        try:
            res = slots_[name]['value']
        except Exception as e:
            res = failed_parse_symbol
    if lower: res = res.lower()
    print('SLOT RESOLVED:', res)
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
