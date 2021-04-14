import os
from pathlib import Path
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


def load_prompt(name) -> str:
    """Load a gpt3 prompt from a text file in data/prompts.

    Parameters
    ----------
    name: str
        Name of file, not counting the extension or preceding directories.

    Returns
    -------
    str: Text in file with start/end spaces stripped.
    """
    return load(f'data/prompts/{name}.txt').strip()


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

