"""General purpose utilities."""

from colorama import Fore
from datetime import datetime, time
from functools import update_wrapper, partial, wraps
from inspect import _empty, Parameter, signature
import json
import logging
import os
from pathlib import Path
import re
import sys
from threading import Thread
import warnings
import yaml
import _thread

from htools import select, bound_args, copy_func, xor_none, add_docstring, \
    listlike, MultiLogger, tolist, load, save
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


def interrupt_on_complete(meth):
    """Decorator that powers PropagatingThread. We do this here rather than
    interrupting directly from run() method because run must return in order
    for thread to stop, but of course we can't do anything directly from the
    method after returning.
    """
    @wraps(meth)
    def wrapper(*args, **kwargs):
        res = meth(*args, **kwargs)
        if args[0].exception and args[0].raise_immediately:
            print('INTERRUPTING')
            _thread.interrupt_main()
        return res
    return wrapper


class PropagatingThread(Thread):
    """Ported from gui/utils.py since I realized ReturningThread never raises
    error to calling context if it fails.

    Thread that will raise an exception in the calling thread as soon as one
    occurs in the worker thread. You must use a KeyboardInterrupt or
    BaseException in your try/except block since that is the only type of
    exception we can raise. If you don't need the exception to be raised
    immediately, you can set raise_immediately=False and it won't be propagated
    until the thread is joined.

    thread.join() also returns the value returned by the thread's target
    function.

    Partially based on version here, though that can only raise on join():
    https://stackoverflow.com/questions/2829329/catch-a-threads-exception-in-the-caller-thread
    """

    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs=None, *, daemon=None,
                 raise_immediately=False):
        """
        Parameters
        ----------
        raise_immediately: bool
            If True, raise any exception as soon as it occurs. In order to
            propagate this to the calling thread, we are forced to use a
            KeyboardInterrupt. No information about the actual exception will
            be propagated. If False, the actual exception will be raised on
            the call to thread.join().
        """
        super().__init__(group=group, target=target, name=name,
                         args=args, kwargs=kwargs, daemon=daemon)
        self.raise_immediately = raise_immediately
        self.exception = None
        self.result = None

    @interrupt_on_complete
    def run(self):
        try:
            self.result = self._target(*self._args, **self._kwargs)
        except Exception as e:
            self.exception = e

    def join(self, timeout=None):
        super().join(timeout)
        if not self.raise_immediately and self.exception:
            raise self.exception
        return self.result


def thread_starmap(func, kwargs_list=None, raise_errors=True,
                   raise_immediately=False):
    """Similar to multiprocessing.Pool.starmap but with my ReturningThread
    instead.

    Parameters
    ----------
    raise_errors: bool
        If True, any error in one of the underlying threads will result in an
        error being raised to the calling context. If False, no error will be
        raised and the corresponding list item will simply be None.
    raise_immediately: bool
        If raise_errors is False, this is unused. If it's True, it determines
        whether exceptions that occur in threads are raised immediately when
        they happen (if True) or only when the failed thread is joined
        (if False).
    """
    kwargs_list = kwargs_list or [{}]
    if raise_errors:
        thread_cls = partial(PropagatingThread,
                             raise_immediately=raise_immediately)
    else:
        thread_cls = ReturningThread
    threads = [thread_cls(target=func, kwargs=kwargs)
               for kwargs in tolist(kwargs_list)]
    for thread in threads: thread.start()
    return [thread.join() for thread in threads]


class JsonlinesFormatter(logging.Formatter):
    """Formatter for logging python data structures to a jsonlines file.
    Used by JsonLogger.
    """

    def format(self, record):
        return json.dumps(record.msg)


class JsonlinesLogger(MultiLogger):
    fmt = '%(message)s'

    def __init__(self, path):
        super().__init__(path, fmode='a', fmt=self.fmt)
        self.formatter = self._add_json_formatter()
        self.path = self._get_file_handler()[0].baseFilename

    def _log(self, level, msg, args, exc_info=None, extra=None,
             stack_info=False):
        if not os.path.isfile(self.path):
            raise RuntimeError('PATH DOES NOT EXIST.')
        super()._log(level, msg, args, exc_info=exc_info, extra=extra,
                     stack_info=stack_info)

    def _get_file_handler(self):
        """
        Returns
        -------
        tuple: First item is FileHandler, second is its index in the logger's
        file handlers. This allows us to effectively edit it.
        """
        handlers = [(handler, i) for i, handler in enumerate(self.handlers)
                    if isinstance(handler, logging.FileHandler)]
        if len(handlers) != 1:
            raise RuntimeError(
                'Expected JsonlinesLogger to have 1 FileHandler, '
                f'found {len(handlers)}.'
            )
        return handlers[0]

    def _add_json_formatter(self):
        formatter = JsonlinesFormatter(self.fmt)
        handler, _ = self._get_file_handler()
        handler.setFormatter(formatter)
        return formatter

    def change_path(self, path):
        """Change the path of the file we log to by creating a new FileHandler
        because the filepath affects other attributes that are set in the
        constructor.
        """
        _, i = self._get_file_handler()
        self.handlers[i] = self._change_path(self.handlers[i], path)
        self.path = self.handlers[i].baseFilename

    def _change_path(self, handler, path):
        kwargs = {key: getattr(handler, key) for key in
                  ('mode', 'encoding', 'delay')}
        handler = type(handler)(filename=path, **kwargs)
        handler.setFormatter(self.formatter)
        return handler


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


def touch(path):
    """Create an empty file, like bash `touch` command. If necessary, we
    create the parent dir as well.
    """
    os.makedirs(Path(path).parent, exist_ok=True)
    with open(path, 'a'):
        pass


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


def load_api_key(name, raise_error=True):
    """Generic api key loader. Assumes you store the key as a text file
    containing only one key in a file called ~/.{name}.

    Parameters
    ----------
    name: str
        Examples: 'booste', 'huggingface', 'gooseai'.
    raise_error: bool
        If True, a missing file will raise an error. If False, a missing file
        will only cause a warning and will return an empty string.

    Returns
    -------
    str
    """
    def _load_key(name):
        """Load api key or raise FileNotFoundError. We also check for
        an environment variable like OPENAI_API_KEY.
        """
        try:
            with open(Path(f'~/.{name}').expanduser(), 'r') as f:
                return f.read().strip()
        except FileNotFoundError as e:
            key = os.getenv(f'{name.upper()}_API_KEY')
            if key: return key
            raise e

    if raise_error:
        return _load_key(name)
    try:
        return _load_key(name)
    except FileNotFoundError:
        warnings.warn(
            f'Jabberwocky expected to find an api key for backend '
            f'{name} but couldn\'t. Using {name} backend will throw '
            f'auth errors. Add your api key to "~/.{name} (if you are using a '
            f'GPTBackend object, you will then need to use its '
            f'refresh_api_keys method to enable the backend.'
        )
        return ''


load_goose_api_key = partial(load_api_key, name='gooseai')


def save_yaml(data, path):
    """Save a dictionary as a yaml file.

    Parameters
    ----------
    data: dict
    path: str or Path
        Any intermediate dirs that do not already exist will be created.
    """
    path = Path(path)
    if not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        yaml.dump(data, f)


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


def escape_markdown(text):
    raw = r'([!#()*+-.[\]_`{|}])'
    return re.sub(raw, r'\\\1', text)


def _update_prompt_readme(dir_='data/prompts', sep='***'):
    """Update readme file in a directory of prompt yaml files to contain a
    table displaying their name and description (provided by the "doc" field
    in the yaml file.
    """
    dir_ = Path(dir_)
    name2doc = {path.stem: escape_markdown(load_yaml(path).get('doc', '')
                                           .replace('\n', ' '))
                for path in dir_.iterdir() if path.suffix == '.yaml'}
    tbody = '\n'.join(f'{k} | {v}' for k, v in sorted(name2doc.items(),
                                                      key=lambda x: x[0]))
    table = f'Prompt Name | Doc\n---|---\n{tbody}'
    try:
        readme = load(dir_/'README.md')
    except FileNotFoundError:
        readme = sep
    keep, sep, _ = readme.partition(sep)
    readme = f'{keep}{sep}\n{table}'
    save(readme, dir_/'README.md')
    return readme


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


def containerize(*args, dtype=list):
    """Basically applies tolist() to a bunch of objects, except that tolist
    doesn't work on dicts. I considered changing that but I'd have to check
    what went into that decision - I have a vague suspicion that that behavior
    is important to incendio.

    Returns
    -------
    tuple[list]: Each item in the returned tuple is a list-like object. If the
    corresponding input arg was not like-like, it becomes the first value in
    a new list, otherwise we keep the input type (so it could be a tuple, for
    example).
    """
    res = []
    for arg in args:
        if not listlike(arg):
            arg = [arg]
        res.append(dtype(arg))
    return res


def register(name, mapping):
    """Decorator to add a function to some module-level dict mapping a string
    to a function.

    Parameters
    ----------
    name
    mapping

    Examples
    --------
    META = {}

    @register('foo', META)
    def stream_foo(x, y):
        return x * y

    >>> META
    {'foo': <function __main__.stream_foo(x, y, **kwargs)>}
    """
    def decorator(func):
        mapping[name] = func
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    return decorator


def namecase(name):
    """Similar to titlecase but don't force lowercase letters after the first
    one in each word:

    name = 'Yann LeCun'
    name.title()   # 'Yann Lecun'
    namecase(name) # 'Yann LeCun'
    """
    chunks = name.split()
    return ' '.join(chunk[0].upper() + chunk[1:] for chunk in chunks)


def seconds_til_midnight(dt=None):
    """Compute seconds til midnight so we know how long to sleep for when
    changing date-based log file names. See
    jabberwocky.openai_utils.GPTBackend.update_log_path().
    """
    dt = dt or datetime.today()
    midnight = datetime.combine(dt, time())
    return (midnight - dt).seconds