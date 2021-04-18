import os
from pathlib import Path
import yaml
import sys
import warnings

from htools import load
from jabberwocky.config import C


def load_booste_api_key():
    """Load api key for booste, a way to programmatically access gpt2 and clip.

    Returns
    -------
    str
    """
    with open(Path('~/.booste').expanduser(), 'r') as f:
        return f.read().strip()


def load_api_key():
    """Load openai API key. This must either be an environment variable called
    OPENAI_API_KEY or placed in a text file at ~/.openai.

    Returns
    -------
    str
    """
    key = os.getenv('OPENAI_API_KEY')
    if not key:
        with open(Path('~/.openai').expanduser(), 'r') as f:
            key = f.read().strip()
    return key


def openai_auth():
    """Load openai api key and try to set it in the openai library. This must
    be done after importing openai.
    """
    os.environ['OPENAI_API_KEY'] = key = load_api_key()
    try:
        module = sys.modules['openai']
        module.api_key = key
    except Exception as e:
        warnings.warn('openai library has not been imported. API key not set.')


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


def load_prompt(name, prompt='', rstrip=True):
    """Load a gpt3 prompt from data/prompts. Note that this function went
    through several iterations and early versions of this function didn't
    allow for an input prompt parameter. This worked fine for toy examples
    where the prompt is static (e.g. reformatting some dates) but as we get to
    more powerful prompts we often want to specify recommended hyperparameters
    and allow for inputting new text, so a yaml file became more appropriate.
    However, getting new lines, special characters, and brackets (for string
    formatting) to all work in yaml files turns out to be surprisingly hard, so
    we instead place the prompt in its own .txt file and leave the .yaml file
    for hypers.

    Parameters
    ----------
    name: str
        Name of subdirectory in data/prompts. Ex: 'simplify_ml'
    prompt: str
        Additional input to be inserted into the prompt template. For example,
        our tldr template prompt is "{}\n\ntl;dr:". We need to pass in text
        to summarize (this replaces the brackets like in a python f-string).
    rstrip: bool
        This is a safety measure to prevent us from accidentally leaving a
        trailing space after the end of the prompt (which leads to worse gpt3
        completions). We let the user turn it off in case a prompt requires it.

    Returns
    -------
    dict: Keys are all kwargs for query_gpt3(). You may want to override some
    of these at times, but they at least provide reasonable defaults. Some are
    more important than others: for example, a 'stop' value will likely always
    be relevant, while 'max_tokens' or 'engine_i' may depend on the specific
    usage.
    """
    dir_ = Path(f'data/prompts/{name}')
    prompt_fmt = load(dir_/'prompt.txt')
    kwargs = load_yaml(dir_/'config.yaml')
    prompt = prompt_fmt.format(prompt)
    if rstrip: prompt = prompt.rstrip()
    kwargs['prompt'] = prompt
    msg = kwargs.pop('reminder', None)
    if msg: print(msg)
    return kwargs


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


def print_response(prompt, response):
    """Print gpt3 prompt and response. Prompt is in bold to make it easier to
    distinguish between them.

    Parameters
    ----------
    prompt: str
    response: str
        The text returned by gpt3.
    """
    print(bold(prompt), end='')
    print(response)


