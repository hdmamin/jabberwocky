"""Functionality to fetch and work with YouTube transcripts and Wikipedia data.
"""
from bs4 import BeautifulSoup
from fuzzywuzzy import fuzz, process
# This is the first jabberwocky module to import sent_tokenize so we only need
# to do this here. Seems like we can't specify this in requirements.txt, though
# maybe we could do some kind of post-install hook in the future.
try:
    from nltk.tokenize import sent_tokenize
except:
    import nltk
    nltk.download('punkt')
    from nltk.tokenize import sent_tokenize
import numpy as np
import os
import pandas as pd
from pathlib import Path
import re
import requests
import unidecode
import warnings
import wikipedia as wiki
from wikipedia import PageError, DisambiguationError
from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound

from htools import DotDict, tolist, Results
from jabberwocky.utils import namecase


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


def infer_gender(text, eps=1e-6):
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


def infer_nationality(summary, name, page=None, qa_pipe=None,
                      question_fmt='What country is {} from?'):
    """Infer a person's nationality from their wikipedia summary.
    Developed in nb17. Note that with no qa pipeline provided, this defaults to
    an html parsing strategy which may be rather brittle (i.e. if wikipedia
    renames a single css class, it will likely fail).

    Parameters
    ----------
    summary: str
        Wikipedia summary. In practice, the nationality info almost always
        comes in the first sentence or two so we should be fine to use the
        truncated summary in wiki_data(). In initial experiments, the full
        summary slowed things down without noticeably helping performance.
    name: str
        Pretty-formatted name (e.g. 'Albert Einstein', not 'albert_einstein').
    page: None or wikipedia.WikipediaPage
        Must be provided when qa_pipe is None. Ignored otherwise.
    qa_pipe: transformers.QuestionAnsweringPipeline. I believe the default
        model is extractive, not generative. If no pipeline is provided, we
        fall back to an html-parsing strategy.
    question_fmt: str
        This defines the question the pipeline tries to answer. It should
        generally contain 1 unnamed field where we will auto-insert the
        person's name.

    Returns
    -------
    tuple[str, float]: First result is the answer, second is the model
    confidence. If an answer couldn't be retrieved, the answer will be an empty
    string and the confidence will be 0. If no qa_pipe is provided, confidence
    is set to 1.0 since we're using a deterministic method (though in reality
    there may occasionally still be errors).
    """
    if qa_pipe:
        answer = qa_pipe({'question': question_fmt.format(name),
                          'context': summary})
        return answer['answer'], answer['score']

    assert page, 'Page must not be None when no Q/A pipe is provided.'
    soup = BeautifulSoup(page.html(), 'lxml')
    tds = [row for row in soup.find_all('th', class_='infobox-label')
           if row.text == 'Born']
    if not tds:
        return '', 0.
    if len(tds) > 1:
        warnings.warn('Found multiple matching candidates: '
                      f'{[row.text for row in tds]}')
    td = tds[0].findNext('td')
    last = list(td.children)[-1]
    text = getattr(last, 'text', last)
    res = re.sub('\[\d{1,2}\]', '', text.split(', ')[-1]).replace('.', '')
    # Just a hacky fix to make outputs a bit more similar to the QA pipeline
    # format. Not worth using a complex solution here because the whole point
    # is to provide a quick stopgap solution for times when I don't want the
    # QA pipeline import to slow me down, i.e. everytime jupyter autoreloads
    # the jabberwocky.openai_utils module.
    mapping = {'US': 'American',
               'England': 'English',
               'France': 'French',
               'Germany': 'German'}
    return mapping.get(res, res), 1.


def wiki_page(name, *tags, retry=True, min_similarity=50, debug=False,
              og_name=None, auto_suggest=False):
    """Note: on a sample of 27 names, setting auto_suggest=True had nearly a
    50% failure rate. This dropped to <5% when setting it to False (and the 1
    failure works if I provide 1 tag).

    Parameters
    ----------
    name
    tags: str
        Optional: provide string(s) related to the name to help with
        disambiguation. E.g. Sometimes entries are named something like
        "John Smith (athlete)" so it would help to provide "athlete" as a tag.
        This only helps if retry=True.
    retry
    min_similarity
    debug
    og_name: None or str
        Never pass this in explicitly - it's only there for when the function
        recursively calls itself after concatenating tags with the original
        name.
    auto_suggest

    Returns
    -------

    """
    try:
        page = wiki.page(namecase(name), auto_suggest=auto_suggest)
        score = fuzz.token_set_ratio((og_name or name).lower(),
                                     page.title.lower())
        if score < min_similarity:
            raise RuntimeError(
                f'Similarity score of {score} fell short of threshold '
                f'{min_similarity}. Page title: {page.title}.'
            ) from None
        return page
    except PageError:
        # if not retry or not tags:
        if not retry:
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


def cleanup_wiki_name(name):
    """
    'J. K. Rowling' -> 'JK Rowling'
    'A. B. C. Lastname' -> 'ABC Lastname'
    'Ted P. Chang' -> 'Ted P Chang'
    'John Smith (American Wrestler)' - > 'John Smith'
    """
    n = name.count('. ')
    if n > 1:
        name = re.sub('. ', '.', name, count=n - 1)
    name = name.replace('.', '').split('(')[0].strip()
    return unidecode.unidecode(name)


def wiki_data(name, tags=(), img_dir='data/tmp', exts={'jpg', 'jpeg', 'png'},
              fname=None, truncate_summary_lines=2, qa_pipe=None, verbose=True,
              **page_kwargs):
    """Warning: I think name may require title case (may fail for names with
    unusual capitalization though).

    Parameters
    ----------
    name
    tags
    img_dir
    exts
    fname: str or None
        If provided, this should be a file name (no directory name and no
        suffix) for the image to download. Defaults to the person's name
        otherwise.
    truncate_summary_lines
    verbose
    page_kwargs

    Returns
    -------

    """
    try:
        page = wiki_page(name, *tolist(tags), **page_kwargs)
    except RuntimeError as e:
        raise e
    except DisambiguationError as e:
        raise RuntimeError from e
    gender = infer_gender(page.summary)[0]
    summary = page.summary.splitlines()[0]
    if truncate_summary_lines:
        summary = ' '.join(sent_tokenize(summary)[:truncate_summary_lines])
    nationality, _ = infer_nationality(summary, name, page=page,
                                       qa_pipe=qa_pipe)
    # Use page title to avoid misspellings, e.g. I typed "Apollo Ohno" but
    # meant "Apolo Ohno".
    name = cleanup_wiki_name(page.title)

    # Download image if possible. Find photo with name closest to the one we
    # searched for (empirically, this seems to be a decent heuristic to give
    # us a picture of the person rather than of, for instance, their house).
    img_url = ''
    img_path = ''
    if img_dir and page.images:
        name2url = {u.rpartition('/')[-1].split('.')[0].lower(): u
                    for u in page.images if u.rpartition('.')[-1] in exts}
        # Some pages have no images.
        if name2url:
            img_name, _ = process.extractOne(name.lower(), name2url.keys())
            url = name2url[img_name]
            suff = url.rpartition(".")[-1]
            path = Path(img_dir)/f'{fname or name}.{suff}'.lower()
            os.makedirs(img_dir, exist_ok=True)
            if download_image(url, path, verbose, headers=WIKI_HEADERS):
                img_url = url
                img_path = str(path)
    return Results(name=name,
                   summary=_wiki_text_cleanup(summary),
                   img_url=img_url,
                   img_path=img_path,
                   gender=gender,
                   nationality=nationality)
