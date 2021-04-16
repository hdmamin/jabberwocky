from htools.structures import IndexedDict


class C:
    # Formatting
    bold_start = '\033[1m'
    bold_end = '\033[0m'

    # OpenAI constants
    engines = ['ada', 'babbage', 'curie', 'davinci']
    prices = IndexedDict(zip(engines, [.0008, .0012, .006, .06]))

    # Data
    mock_stream_paths = {True: 'data/misc/sample_stream_response.pkl',
                         False: 'data/misc/sample_response.pkl'}


