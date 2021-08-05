"""General purpose utilities."""

from colorama import Fore
from functools import update_wrapper
from inspect import _empty, Parameter, signature
from pathlib import Path
from PIL import Image
import sys
import yaml

from htools import select, bound_args, copy_func, xor_none
from jabberwocky.config import C


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


def load_booste_api_key():
    """Load api key for booste, a way to programmatically access gpt2 and clip.

    Returns
    -------
    str
    """
    with open(Path('~/.booste').expanduser(), 'r') as f:
        return f.read().strip()


def load_huggingface_api_key():
    """Load api used when querying Huggingface model APIs. This will be placed
    in the headers for a post request like
    {'Authorization': f'Bearer api_{my_api_key}'}.
    """
    with open(Path('~/.huggingface').expanduser(), 'r') as f:
        return f.read().strip()


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
