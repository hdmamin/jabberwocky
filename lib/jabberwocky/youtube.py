from fuzzywuzzy import fuzz
import numpy as np
import pandas as pd
import re
from youtube_transcript_api import YouTubeTranscriptApi

from htools import DotDict


def text_segment(df, start, end):
    """Return text corresponding to a user-specified time segment.

    Parameters
    ----------
    df: pd.DataFrame
        Results of youtube_transcript_api cast to a df.
    start: int
        Start time in seconds.
    end: int
        End time in seconds. The duration will therefore be (at least)
        (end - start) seconds.

    Returns
    -------
    str: Text transcript of the speech during the specified time segment. May
    include some text from before/after the specified segment because text is
    given in chunks of a few seconds (i.e. if we ask for time=3s to time=10s,
    we might end up with time=2s to time=13s because we don't have more
    fine-grained information).
    """
    assert end > start, 'End time must be later than start time.'

    if start < df.start.iloc[0]:
        start_idx = 0
    else:
        start_idx = df.loc[df.start <= start].index[-1]

    if end > df.start.iloc[-1]:
        end_idx = df.tail(1).index[0]
    else:
        end_idx = df.loc[df.start >= end].index[0]
    return ' '.join(df.iloc[start_idx:end_idx+1].text)


def video_id(url):
    """Extract a video ID from an youtube video URL. Not super strict about
    validation: we assume you'll pass something halfway reasonable.

    Parameters
    ----------
    url: str
        Example: https://www.youtube.com/watch?v=9JpdAg6uMXs
        Notice there isn't any channel information appended after the ID.

    Returns
    -------
    str: Video ID, e.g. 9JpdAg6uMXs for the example in Parameters.
    """
    parts = url.split('watch?v=')
    assert len(parts) == 2, 'Unrecognized url format. URL should look like ' \
                            'https://www.youtube.com/watch?v=asjasdjfh'
    return parts[-1]


def get_transcripts(url, verbose=True):
    """Fetch one or more transcripts for a youtube video given its URL.

    Parameters
    ----------
    url: str

    Returns
    -------
    DotDict: Contains keys 'id' (maps to video ID str), 'generated',
    and 'manual' (the latter two lap to pandas dfs or None if no transcript was
    found). Manual transcripts are human-created. Generated transcripts are a
    bit lower quality and tend to lack punctuation.
    """
    id_ = video_id(url)
    types = {True: 'generated', False: 'manual'}
    results = dict.fromkeys(types.values())
    for trans in YouTubeTranscriptApi.list_transcripts(id_):
        if trans.language_code == 'en':
            rows = trans.fetch()
            results[types[trans.is_generated]] = pd.DataFrame(rows)
    if verbose:
        null_keys = [k for k, v in results.items() if v is None]
        print(f'The following keys are null: {repr(null_keys)}')
    return DotDict(**results, id=id_)


def realign_punctuated_text(df, text, skip_1st=1, margin=2):
    """Realign gpt3's punctuated version of a youtube transcript with the
    original timestamps.

    Parameters
    ----------
    df: pd.DataFrame
        Auto-generated transcript from YouTubeTranscriptApi. Contains 'text',
        'start', and 'duration' columns.
    text: str
        Punctuated text returned by GPT3.
    skip_1st: int
        Number of words at the start of each punctuated chunk to skip. The
        minimum number of words per row will be skip_1st+2. This helps protect
        slightly against cases where the true line is a common sequence that
        we encounter at the very start of the candidate
        Ex:
        unpunctuated: 'it is what it is'
        punctuated: 'It is what it is.'

        With skip_1st=0, we may mistakenly select "It is" from the punctuated
        candidate. With skip_1st=1, we select "It is what it is." as desired.
    margin: int
        Number of additional words to consider when constructing a punctuated
        chunk (list) from our punctuated text.

        Ex (margin=2):
        unpunctuated: ['this', 'is', 'true', 'however', 'we']
        punctuated: ['This', 'is', 'true', '-', 'however', 'we', 'still']
        Notice that adding punctuation can create new tokens, so with margin=0
        our candidate would be truncated prematurely (ending with 'however').
    """
    # Built-in str.split doesn't retain starting/trailing spaces correctly.
    # Probably would be fine but just keep this since it took a while to get
    # right and I don't want to break it.
    words = re.split(' ', text)
    rows = []
    start_i = 0
    for i, chunk in df.iterrows():
        chunk_words = re.split(' ', chunk.text)
        length = len(chunk_words)
        punct_words = words[start_i:start_i + length + margin]
        suff = ' '.join(chunk_words[-2:])
        scores = []
        bigrams = zip(punct_words[skip_1st:], punct_words[skip_1st + 1:])
        # Avoid list comp so we can exit early if we find a perfect match.
        for i, gram in enumerate(bigrams):
            score = fuzz.ratio(suff, ' '.join(gram))
            if score == 100:
                argmax = i
                break
            scores.append(score)
        else:
            argmax = np.argmax(scores)
        punct_len = skip_1st + argmax + 2
        rows.append(' '.join(words[start_i:start_i + punct_len]))
        start_i += punct_len

    new_df = pd.DataFrame(rows, columns=['text'])
    return pd.concat((new_df, df.reset_index()[['start', 'duration']].copy()),
                     axis=1)
