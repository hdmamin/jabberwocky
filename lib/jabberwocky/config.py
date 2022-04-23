"""Define constants used throughout the project.
"""

from pathlib import Path
import warnings

from htools.structures import IndexedDict


class C:
    # Formatting (note: later realized bold_end actually also ends colors too.)
    bold_start = '\033[1m'
    underline_start = '\033[4m'
    bold_end = '\033[0m'

    # OpenAI constants
    # Parameter Counts: 2.7B, 6.7B, 13B, 175B
    engines = ['text-ada-001',
               'text-babbage-001',
               'text-curie-001',
               'text-davinci-002']
    backend_engines = {
        'openai': engines,
        'gooseai': ['gpt-neo-2-7b', 'gpt-j-6b', 'fairseq-13b', 'gpt-neo-20b'],
        'huggingface': ['gpt-neo-125M', 'gpt-neo-1.3B', 'gpt-neo-2.7B',
                        'gpt-j-6B'],
        # These backends only provide 1 model so these are just filler names.
        # (They give GPTBackend.engine() something to return but they don't
        # actually affect the query results.)
        'hobby': ['gpt-j-6B' for _ in range(4)],
        'banana': ['gpt-j-6B' for _ in range(4)]
    }

    # Dollars per thousand tokens with openai backend.
    prices = IndexedDict(zip(engines, [.0008, .0012, .006, .06]))

    # Data
    root = Path(__file__).parent.parent.parent
    if not (root.stem == 'jabberwocky'
            and 'data' in {p.stem for p in root.iterdir()}):
        warnings.warn('Jabberwocky.config.root does not match the expected '
                      'project root. If you haven\'t cloned the repo from '
                      'github, you probably should.')

    mock_stream_paths = {True: root/'data/misc/sample_stream_response.pkl',
                         False: root/'data/misc/sample_response.pkl'}
    # This contains a dict generated by scripts/s01_fetch_sample_responses.py.
    # Contains more variants of responses (see script docstring for details).
    all_mocks_path = root/'data/misc/gooseai_sample_responses.pkl'

    # Project root directory.
    root = Path('/Users/hmamin/jabberwocky')

    # Use with datetime.strptime() to convert from format like
    # 'Sun Apr 17 13:18:23 2022' back to datetime object. This is the format
    # GPTBackend.query logger uses in the meta.datetime field.
    ctime_fmt = '%a %b %d %H:%M:%S %Y'

