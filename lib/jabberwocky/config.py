"""Define constants used throughout the project.
"""

from pathlib import Path

from htools.structures import IndexedDict


class C:
    # Formatting (note: later realized bold_end actually also ends colors too.)
    bold_start = '\033[1m'
    underline_start = '\033[4m'
    bold_end = '\033[0m'

    # OpenAI constants
    # Parameter Counts: 2.7B, 6.7B, 13B, 175B
    engines = ['ada', 'babbage', 'curie', 'davinci']
    backend_engines = {
        'openai': engines,
        'gooseai': ['gpt-neo-2-7b', 'gpt-j-6b', 'fairseq-13b', 'gpt-neo-20b'],
        'huggingface': ['gpt-neo-125M', 'gpt-neo-1.3B', 'gpt-neo-2.7B',
                        'gpt-j-6B'],
        # Filler names.
        'hobby': ['', '', '', '']
    }

    prices = IndexedDict(zip(engines, [.0008, .0012, .006, .06]))

    # Data
    root = Path(__file__).parent.parent.parent
    mock_stream_paths = {True: root/'data/misc/sample_stream_response.pkl',
                         False: root/'data/misc/sample_response.pkl'}

