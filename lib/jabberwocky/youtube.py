import pandas as pd
import warnings
from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound

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
        Don't include any channel-related suffix. E.G. use
        https://www.youtube.com/watch?v=OZbCRN3C_Hs, not
        https://www.youtube.com/watch?v=OZbCRN3C_Hs&ab_channel=BBC.
    verbose: bool
        Warn

    Returns
    -------
    DotDict: Contains keys 'id' (maps to video ID str), 'generated',
    and 'manual' (the latter two lap to pandas dfs or None if no transcript was
    found). Manual transcripts are human-created. Generated transcripts are a
    bit lower quality and tend to lack punctuation.
    """
    langs = ['en', 'en-GB']
    id_ = video_id(url)
    res = {'generated': None, 'manual': None}
    transcripts = YouTubeTranscriptApi.list_transcripts(id_)
    res['generated'] = transcripts.find_generated_transcript(langs)
    try:
        res['manual'] = transcripts.find_manually_created_transcript(langs)
    except NoTranscriptFound:
        if verbose: warnings.warn('No manual transcript found.')
    if verbose:
        non_en_keys = [k for k, v in res.items()
                       if v and ('United Kingdom' in v.language)]
        if non_en_keys:
            warnings.warn(
                f'{non_en_keys} {"has" if len(non_en_keys) == 1 else "have"} '
                'language en-GB, not en.'
            )
    return DotDict(
        **{k: pd.DataFrame(v.fetch()) if v else v for k, v in res.items()},
        id=id_
    )


