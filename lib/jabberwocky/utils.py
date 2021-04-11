import os
from pathlib import Path

from htools import load
from jabberwocky.config import C


def load_api_key() -> str:
    """Load openai API key. This must either be an environment variable called
    OPENAI_API_KEY or placed in a text file at ~/.openai.
    """
    key = os.getenv('OPENAI_API_KEY')
    if not key:
        with open(Path('~/.openai').expanduser(), 'r') as f:
            key = f.read().strip()
    return key


def openai_auth():
    """
    # TODO: docs
    Returns
    -------

    """
    os.environ['OPENAI_API_KEY'] = load_api_key()


def load_prompt(name) -> str:
    """
    # TODO: docs

    Parameters
    ----------
    name

    Returns
    -------

    """
    return load(f'data/prompts/{name}.txt').strip()


def bold(text):
    """
    # TODO: docs
    Parameters
    ----------
    text

    Returns
    -------

    """
    return C.BOLD_START + text + C.BOLD_END


def print_response(prompt, response):
    print(bold(prompt))
    print(response)

