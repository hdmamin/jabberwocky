"""Define constants used throughout the project.
"""


from htools.structures import IndexedDict


class C:
    # Formatting (note: later realized bold_end actually also ends colors too.)
    bold_start = '\033[1m'
    underline_start = '\033[4m'
    bold_end = '\033[0m'

    # OpenAI constants
    engines = ['ada', 'babbage', 'curie', 'davinci']
    engines_goose = ['gpt-neo-2-7b', 'gpt-j-6b', 'fairseq-13b', 'gpt-neo-20b']
    engines_neo = ['gpt-neo-125M', 'gpt-neo-1.3B', 'gpt-neo-2.7B', 'gpt-j-6B']
    backend_engines = {'openai': engines,
                       'gooseai': engines_goose}

    prices = IndexedDict(zip(engines, [.0008, .0012, .006, .06]))

    # Data
    mock_stream_paths = {True: 'data/misc/sample_stream_response.pkl',
                         False: 'data/misc/sample_response.pkl'}

