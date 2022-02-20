from collections import Mapping

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
    """
    # TODO: docs

    state.set(scope, max_tokens=6)
    X state[scope].max_tokens = 6
    X state[scope]['max_tokens'] = 6
    X state[scope, 'max_tokens'] = 6
    X state.global.max_tokens = 6

    state['max_tokens']
    state.max_tokens
    """
    def __init__(self, global_=None, person_=None, conversation_=None):
        self._global = dict(global_) or {}
        self._person = dict(person_) or {}
        self._conversation = dict(conversation_) or {}

        # Cleaner interface than using getattr all the time.
        self._states = {
            'global': self._global,
            'person': self._person,
            'conversation': self._conversation
        }
        self.state = DotDict()
        self._resolve_state()

    @classmethod
    def clone(cls, settings):
        return cls(settings._global.copy(),
                   settings._person.copy(),
                   settings._conversation.copy())

    def _resolve_state(self):
        # Order matters here: we want global settings to take priority over
        # person-level settings, and both of those to take priority over
        # conversation level settings.
        self.state = DotDict({
            **self._states['conversation'],
            **self._states['person'],
            ** self._states['global']
        })

    def __getitem__(self, key):
        return self.state[key]

    def __iter__(self):
        return iter(self.state)

    def __len__(self):
        return len(self.state)

    # TODO: maybe redo all logic to make getitem/setitem/delitem work with
    # multiple dispatch or passing in a tuple. This does work (even w/out
    # parentheses) but we can't use keyword args.
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

    def __repr__(self):
        return f'Settings({self.state})'


def slot(request, name, lower=True):
    """
    # TODO: docs

    Parameters
    ----------
    request
    name
    lower
    """
    failed_parse_symbol = '?'
    slots_ = request['request']['intent']['slots']
    if name in slots_:
        print('SLOTS', slots_)
        try:
            resolved = slots_[name]['resolutions']['resolutionsPerAuthority']
            res = resolved[0]['values'][0]['value']['name']
        except KeyError:
            # Some intents have optional slots. If the user excludes one slot,
            # 'resolutions' will be missing.
            res = failed_parse_symbol
    else:
        res = list(slots_.values())[0]['value']
    if lower: res = res.lower()
    return '' if res == failed_parse_symbol else res
