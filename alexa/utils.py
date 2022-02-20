from htools.meta import delegate
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


class Settings:
    """
    # TODO: docs
    """
    def __init__(self, sess):
        # print('sess', sess)
        # self._attrs = sess.attributes
        self._states = {
            'global': {},
            'person': {},
            'conversation': {}
        }
        self.state = DotDict()

    def resolve_states(self):
        # TODO
        pass

    def __getitem__(self, key):
        return self.state[key]

    # TODO: maybe redo all logic to make getitem/setitem/delitem work with
    # multiple dispatch or passing in a tuple. This does work (even w/out
    # parentheses) but we can't use keyword args.
    def get(self, key, default=None):
        return self.state.get(key, default=default)

    def __setitem__(self, key, val):
        raise RuntimeError('You cannot use __setitem__ on a Settings object. '
                           'Instead, use the `set` method.')

    def __delitem__(self, key):
        raise RuntimeError('You cannot use __delitem__ on a Settings object. '
                           'Instead, use the `delete` method.')

    def set(self, key, val, level):
        self._states[level][key] = val

    def delete(self, key, level):
        del self._states[level][key]

    def __repr__(self):
        return f'SessionState({self.state})'


def slot(request, name, lower=True):
    """
    # TODO: docs

    Parameters
    ----------
    request
    name
    lower

    Returns
    -------

    """
    slots_ = request['request']['intent']['slots']
    if name in slots_:
        resolved = slots_[name]['resolutions']['resolutionsPerAuthority']
        res = resolved[0]['values'][0]['value']['name']
    else:
        res = list(slots_.values())[0]['value']
    if lower: res = res.lower()
    return '' if res == '?' else res
