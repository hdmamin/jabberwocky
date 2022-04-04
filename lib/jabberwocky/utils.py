"""General purpose utilities."""

from colorama import Fore
from functools import update_wrapper, partial
from inspect import _empty, Parameter, signature
from itertools import cycle
from pathlib import Path
from PIL import Image
import sys
from threading import Thread
import yaml

from htools import select, bound_args, copy_func, xor_none, add_docstring, \
    listlike
from jabberwocky.config import C


class ReturningThread(Thread):

    @add_docstring(Thread)
    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs=None, *, daemon=None):
        """This is identical to a regular thread except that the join method
        returns the value returned by your target function. The
        Thread.__init__ docstring is shown below for the sake of convenience.
        """
        super().__init__(group=group, target=target, name=name,
                         args=args, kwargs=kwargs, daemon=daemon)
        self.result = None

    def run(self):
        self.result = self._target(*self._args, **self._kwargs)

    def join(self, timeout=None):
        super().join(timeout)
        return self.result


def with_signature(to_f, keep=False):
    """Decorator borrowed from fastai and renamed to avoid name collision
    with htools (originally called "delegates"). Replaces `**kwargs`
    in signature with params from `to`. Unlike htools.delegates, it only
    changes documentation - variables are still made available in the decorated
    function as 'kwargs'.
    """
    def _f(f):
        from_f = f
        sig = signature(from_f)
        sigd = dict(sig.parameters)
        k = sigd.pop('kwargs')
        s2 = {k: v for k, v in signature(to_f).parameters.items()
              if v.default != Parameter.empty and k not in sigd}
        sigd.update(s2)
        if keep: sigd['kwargs'] = k
        from_f.__signature__ = sig.replace(parameters=sigd.values())
        return f
    return _f


def strip(text, do_strip=True):
    """Convenience function used in query_gpt3.

    Parameters
    ----------
    text: str
    do_strip: bool

    Returns
    -------
    str: If do_strip is False, this is the same as the input.
    """
    return text.strip() if do_strip else text


def squeeze(*args, n=1):
    """Return either the input `args` or the first item of each arg, depending
    on our choice of n. We effectively treat n as a boolean. This is used by
    the query_gpt_{} functions to return either a (str, dict) tuple or a
    (list[str], list[dict]) tuple, depending on the number of completions
    we ask for. If first arg is not some kind of list/tuple (e.g. a str), we
    simply return the args unchanged.
    """
    if not listlike(args[0]):
        return args
    return tuple(arg[0] for arg in args) if n == 1 else args


def load_booste_api_key():
    """Load api key for booste, a way to programmatically access gpt2 and clip.

    Returns
    -------
    str
    """
    raise DeprecationWarning('Booste.ai no longer exists. It was resurrected '
                             '(or at least rebranded) as free.banana.dev. '
                             'Try using `load_banana_api_key`.')
    return load_api_key('booste')


def load_huggingface_api_key():
    """Load api used when querying Huggingface model APIs. This will be placed
    in the headers for a post request like
    {'Authorization': f'Bearer api_{my_api_key}'}.
    """
    return load_api_key('huggingface')


def load_api_key(name):
    """Generic api key loader. Assumes you store the key as a text file
    containing only one key in a file called ~/.{name}.

    Parameters
    ----------
    name: str
        Examples: 'booste', 'huggingface', 'gooseai'.

    Returns
    -------
    str
    """
    with open(Path(f'~/.{name}').expanduser(), 'r') as f:
        return f.read().strip()


load_goose_api_key = partial(load_api_key, name='gooseai')


def load_yaml(path, section=None):
    """Load a yaml file. Useful for loading prompts.

    Parameters
    ----------
    path: str or Path
    section: str or None
        I vaguely recall yaml files can define different subsections. This lets
        you return a specific one if you want. Usually leave as None which
        returns the whole contents.

    Returns
    -------
    dict
    """
    with open(path, 'r') as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    return data.get(section, data)


def bold(text):
    """Make text bold when printed. Note that without the print statement, this
    will just add a weird character string to the front and end of the input.

    Parameters
    ----------
    text: str

    Returns
    -------
    str
    """
    return C.bold_start + text + C.bold_end


def underline(text):
    """Make text underlined when printed.

    Parameters
    ----------
    text: str

    Returns
    -------
    str
    """
    return C.underline_start + text + C.bold_end


def colored(text, color):
    """Update string so it will show up in a different color when printed. Note
    that the raw string itself will now have special characters added to the
    beginning and end, so the repr will look a bit odd.

    Parameters
    ----------
    text: str
    color: str
        Must be a color provided by colorama.Fore.

    Returns
    -------
    str
    """
    return getattr(Fore, color.upper()) + text + Fore.RESET


def getindex(arr, val, default=-1):
    return arr.index(val) if val in arr else default


def most_recent_filepath(dir_, mode='m'):
    """Get path of most recently modified file in a directory.

    Parameters
    ----------
    dir_: str or Path
        Directory to look in.
    mode: str
       One of ('m', 'c') corresponding to mtime or ctime. ctime changes when
       file permissions change while mtime does not, but they're very similar
       otherwise.

    Returns
    -------

    """
    paths = [path for path in Path(dir_).iterdir() if path.is_file()]
    if not paths:
        raise RuntimeError(f'No files in directory {dir_}.')
    return max(paths, key=lambda x: getattr(x.stat(), f'st_{mode}time'))


def _img_dims(curr_width, curr_height, width=None, height=None):
    xor_none(width, height)
    if width:
        width, height = width, int(curr_height * width/curr_width)
    else:
        height, width = height, int(curr_width * height/curr_height)
    return dict(width=width, height=height)


def img_dims(path, width=None, height=None, verbose=False):
    """Given the path to an image file and 1 desired dimension, compute the
    other dimensions that would maintain its height:width ratio.

    Parameters
    ----------
    path: str or Path
    width: int or None
        Desired width of output image. Specify either this OR height.
    height: int or None
        Desired height of output image. Specify either this OR width.
    verbose: bool

    Returns
    -------
    dict
    """
    curr_width, curr_height = Image.open(path).size
    if verbose: print(f'width: {curr_width}, height: {curr_height}')
    return _img_dims(curr_width, curr_height, width=width, height=height)


def set_module_global(module, key, value):
    """Create global variable in an imported module. This is a slightly hacky
    workaround that solves some types of circular imports.

    Parameters
    ----------
    module: str
    key: str
    value: any
    """
    setattr(sys.modules[module], key, value)


def set_module_globals(module, **kwargs):
    """Set multiple global variables in an imported module."""
    for k, v in kwargs.items():
        set_module_global(module, k, v)


def hooked_generator(gen, *hooks):
    """Attach hook(s) to a generator. Motivation: want to be able to return
    query_gpt3 response in Conversationmanager even if stream=True and still
    ensure that updates are made to the conversation history.

    Parameters
    ----------
    gen
    hooks: Iterable[function]
        1 or more functions that accept three arguments: the first is
        the object yielded by the generator at each step, the second is the
        index (starting at zero) of the current step, and the third is a
        boolean value `is_post` specifying whether we're calling it after the
        generator has finished executing. Any return value is
        ignored - it simply executes. If multiple hooks are provided, they will
        be executed in the order they were passed in.

    Yields
    ------
    Values are yielded from the input generator. They will be unchanged unless
    they are mutable AND one or more hooks alter them.
    """
    for i, val in enumerate(gen):
        for hook in hooks:
            hook(val, i)
        yield val
    for hook in hooks:
        hook(val, i, is_post=True)


class Partial:
    """More powerful (though also potentially more fragile) version of
    functools.partial that updates the resulting signature to work better with
    Jupyter's quick documentation feature. We also update __repr__, __str__,
    and __name__ attributes (optionally renaming the source function). Unlike
    functools.partial, we also reorder parameters so that those without
    defaults always come before those with defaults.

    Note: the resulting object is actually a callable class, not a function.
    """

    def __init__(self, func, name=None, **kwargs):
        """
        Parameters
        ----------
        func: function
        name: str or None
            If None, the source function's name will be used.
        kwargs: any
            Default arguments to set, like in functools.partial.
        """
        self.func = copy_func(func)
        self.old_name = func.__name__

        # Track names of positional args in old function since this affects
        # the order args must be passed in if var_positional parameters
        # (*args) are present.
        self.old_pos_pars = []
        self.kwargs_name = ''
        self.args_name = ''
        new_pars = []
        old_sig = signature(self.func)
        for k, v in old_sig.parameters.items():
            # Check parameter kind for error handling and argument resolution
            # in __call__.
            if v.kind == 0:
                raise NotImplementedError(
                    'rigorous_partial does not support functions with '
                    'positional only parameters.'
                )
            elif v.kind == 2:
                self.args_name = k
            elif v.kind == 4:
                self.kwargs_name = k
                break

            if v.kind <= 2:
                self.old_pos_pars.append(k)

            # Assign default value from newly specified kwargs if provided.
            if k in kwargs:
                default = kwargs.pop(k)
                kind = 3
            else:
                default = v.default
                kind = v.kind
            param = Parameter(k, kind, default=default)
            new_pars.append(param)

        # Remaining kwargs only: those that were not present in func's
        # signature. Require that they be keyword only since ordering can
        # cause issues (updating signature affects what we see but doesn't
        # seem to affect the actual order args are passed in, presumably due
        # to old __code__ object).
        for k, v in kwargs.items():
            param = Parameter(k, 3, default=v)
            new_pars.append(param)
        if self.kwargs_name:
            new_pars.append(Parameter(self.kwargs_name, 4))

        # Ensure we don't accidentally place any parameters with defaults
        # ahead of those without them. Third item in tuple is a tiebreaker
        # (defaults to original function's parameter order).
        old_names = [p for p in old_sig.parameters]
        new_pars.sort(
            key=lambda x: (x.kind, x.default != _empty,
                           getindex(old_names, x.name, float('inf')))
        )

        # I honestly forget why we need to set the attribute on self.func too,
        # I just remember it was needed to resolve a bug (I think it was
        # related to *args resolution).
        self.__signature__ = self.func.__signature__ = old_sig.replace(
            parameters=new_pars
        )
        self.__defaults__ = tuple(p.default for p in new_pars if p.kind < 3
                                  and p.default != _empty)
        self.__kwdefaults__ = {p.name: p.default for p in new_pars
                               if p.kind == 3}
        if name: self.func.__name__ = name
        update_wrapper(self, self.func)

    def __call__(self, *args, **new_kwargs):
        # Remember self.func's actual code is unchanged: we updated how its
        # signature appears, but that doesn't affect the actual mechanics.
        # Therefore, we need to carefully resolve args and kwargs so that the
        # function is called so that behavior matches what we'd expect based
        # on the order shown in the signature.
        tmp_kwargs = bound_args(self.func, args,
                                {**self.__kwdefaults__, **new_kwargs})
        final_args = {name: tmp_kwargs.pop(name)
                      for name in self.old_pos_pars}
        final_star_args = final_args.pop(self.args_name, [])
        final_kwargs = select(tmp_kwargs, drop=list(final_args))
        return self.func(*final_args.values(), *final_star_args,
                         **final_kwargs)

    def __repr__(self):
        """Note: the memory address here points to that of the copy of the
        source function stored in self.func.
        """
        return repr(self.func).replace(self.old_name, self.__name__)

    def __str__(self):
        return str(self.func).replace(self.old_name, self.__name__)


def stream_words(text):
    """Like stream_chars but splits on spaces. Realized stream_chars was a bad
    idea because we risk giving SPEAKER turns like
    "This is over. W" and "hat are you doing next?", neither of which would be
    pronounced as intended. We yield with a space for consistency with the
    other streaming interfaces which require no further postprocessing.
    """
    for word in text.split(' '):
        yield word + ' '


def stream_response(text, full):
    """Generator used to stream gpt completions for backends that don't
    natively support it.

    Parameters
    ----------
    text: str
    full: dict

    Returns
    --------
    >>> text, full = query_gpt_huggingface()
    >>> text
    "It is hot"
    >>> full
    {'generated_text': 'It is hot'}

    # Notice we don't change the full response - there's no consistent
    # pattern all the non-streaming backends share.
    >>> for t, f in stream_response(text, full):
    >>>     print(repr(t), f)
    'It ' {'generated_text': 'It is hot'}
    'is ' {'generated_text': 'It is hot'}
    'hot' {'generated_text': 'It is hot'}
    """
    yield from zip(stream_words(text), cycle([full]))