"""Functionality to fetch and work with YouTube transcripts."""

from fuzzywuzzy import fuzz, process
from nltk.tokenize import sent_tokenize
import numpy as np
import os
import pandas as pd
from pathlib import Path
import re
import requests
import warnings
import wikipedia as wiki
from wikipedia import PageError, DisambiguationError
from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound

from htools import DotDict, tolist, Results


WIKI_HEADERS = {'User-Agent': 'http://www.github.com/hdmamin/jabberwocky'}


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


def _wiki_text_cleanup(text):
    """Clean up first paragraph of wikipedia summary.
    Main Steps:
    1. Remove inline citations, e.g. '[1]'.
    2. Remove noisy pronunciation-related text that often follows name, e.g.
    'bə-RAHK hoo-SAYN oh-BAH-mə;' in the below example:
    'Barack Hussein Obama II (bə-RAHK hoo-SAYN oh-BAH-mə; born August 4, 1961)'
    3. Remove repeated spaces (seem common in the wikipedia py package I'm
    using).

    Parameters
    ----------
    text: str
        First paragraph of wikipedia summary about a person. Only tested on
        real people - for a fictional person without a birthdate, this might
        not work as well.

    Returns
    -------
    str
    """
    # Remove inline citations. We also replace an en dash with a more standard
    # dash because dearpygui can't render the former.
    text = re.sub('\[\d*\]', '', text.replace('–', '-'))
    match = re.search('\(.*\)', text)
    if match:
        match = match.group()
        match_parts = [x for x in match.split(';') if x]
        if len(match_parts) > 1:
            text = text.replace(match, '(' + match_parts[-1].strip())
    return re.sub('\s{2,}', ' ', text)


def _infer_gender(text, eps=1e-6):
    """Infer gender from a piece of text by counting pronouns, intended to be a
    person's wikipedia summary. I'm sure there are better ways to do this but
    this is a nice quick method that seems to work pretty well.

    Parameters
    ----------
    text: str
        Usually the summary attribute of a Page object returned by wiki_page().
    eps: float
        Used for smoothing to avoid dividing by zero. This means our returned
        probability will top out at something like .999 rather than 1.0.

    Returns
    -------
    tuple[str, float]: First item is a str in ('M', 'F') corresponding to a
    prediction of male or female. Second item is a smoothed probability
    quantifying our confidence in the first item.
    """
    toks = text.lower().split(' ')
    male_counts = dict.fromkeys(['he', 'him', 'his'], 0)
    female_counts = dict.fromkeys(['she', 'her', 'hers'], 0)
    for tok in toks:
        if tok in male_counts:
            male_counts[tok] += 1
        elif tok in female_counts:
            female_counts[tok] += 1
    counts = [sum(male_counts.values()), sum(female_counts.values())]
    genders = ['M', 'F']
    idx = np.argmax(counts)
    return genders[idx], (counts[idx]+eps) / (sum(counts) + 2*eps)


def wiki_page(name, *tags, retry=True, min_similarity=50, debug=False,
              og_name=None):
    try:
        page = wiki.page(name, auto_suggest=False)
        score = fuzz.token_set_ratio((og_name or name).lower(),
                                     page.title.lower())
        if score < min_similarity:
            raise RuntimeError(
                f'Similarity score of {score} fell short of threshold '
                f'{min_similarity}. Page title: {page.title}.'
            ) from None
        return page
    except PageError:
        if not retry or not tags:
            raise RuntimeError(f'Couldn\'t find wikipedia page for {name}.') \
                from None
        warnings.warn('Page not found. Trying to auto-select correct match.')

        terms = ' '.join(name.split() + list(tags))
        matches = wiki.search(terms)
        if debug: print('matches:', matches)
        for match in matches:
            if '(disambiguation)' in match: continue
            return wiki_page(match, retry=False, og_name=name)


def download_image(url, out_path, verbose=False, **request_kwargs):
    """Ported from spellotape. Given a URL, fetch an image and download it to
    the specified path.

    Parameters
    ----------
    url: str
        Location of image online.
    out_path: str
        Path to download the image to.
    verbose: bool
        If True, prints a message alerting the user when the image could not
        be retrieved.

    Returns
    -------
    bool: Specifies whether image was successfully retrieved.
    """
    try:
        with requests.get(url, stream=True, timeout=10, **request_kwargs) as r:
            if r.status_code != 200:
                if verbose: print(f'STATUS CODE ERROR: {url}')
                return False

            # Write bytes to file chunk by chunk.
            with open(out_path, 'wb') as f:
                for chunk in r.iter_content(256):
                    f.write(chunk)

    # Any time url cannot be accessed, don't care about exact error.
    except Exception as e:
        if verbose: print(e)
        return False

    return True


def wiki_data(name, tags=(), img_dir='data/tmp', exts={'jpg', 'jpeg', 'png'},
              fname=None, truncate_summary_lines=2, verbose=True,
              **page_kwargs):
    try:
        page = wiki_page(name, *tolist(tags), **page_kwargs)
    except RuntimeError as e:
        raise e
    except DisambiguationError as e:
        raise RuntimeError from e
    gender = _infer_gender(page.summary)[0]
    summary = page.summary.splitlines()[0]
    if truncate_summary_lines:
        summary = ' '.join(sent_tokenize(summary)[:truncate_summary_lines])

    # Download image if possible. Find photo with name closest to the one we
    # searched for (empirically, this seems to be a decent heuristic to give
    # us a picture of the person rather than of, for instance, their house).
    img_url = ''
    img_path = ''
    if img_dir and page.images:
        name2url = {u.rpartition('/')[-1].split('.')[0].lower(): u
                    for u in page.images if u.rpartition('.')[-1] in exts}
        name, _ = process.extractOne(name.lower(), name2url.keys())
        url = name2url[name]
        suff = url.rpartition(".")[-1]
        path = Path(img_dir)/f'{fname or name}.{suff}'.lower()
        os.makedirs(img_dir, exist_ok=True)
        if download_image(url, path, verbose=verbose, headers=WIKI_HEADERS):
            img_url = url
            img_path = str(path)
    return Results(summary=_wiki_text_cleanup(summary),
                   img_url=img_url,
                   img_path=img_path,
                   gender=gender)
