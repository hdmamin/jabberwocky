import os
from pathlib import Path

from htools.structures import IndexedDict


ENGINES = ['ada', 'babbage', 'curie', 'davinci']
PRICES = IndexedDict(zip(ENGINES, [.0008, .0012, .006, .06]))


def load_api_key() -> str:
    """Load openai API key. This must either be an environment variable called
    OPENAI_API_KEY or placed in a text file at ~/.openai.
    """
    key = os.getenv('OPENAI_API_KEY')
    if not key:
        with open(Path('~/.openai').expanduser(), 'r') as f:
            key = f.read().strip()
    return key

