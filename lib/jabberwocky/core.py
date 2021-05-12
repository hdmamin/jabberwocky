"""Core functionality that ties together multiple APIs."""

from fuzzywuzzy import fuzz
import numpy as np
import pandas as pd
import re
from string import punctuation
import warnings
from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound

from htools import flatten, ifnone, Args, auto_repr
from jabberwocky.openai_utils import punctuate_mock_func, PromptManager
from jabberwocky.youtube import video_id


def realign_punctuated_text(df, text, skip_1st=0, margin=2):
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
        for j, gram in enumerate(bigrams):
            score = fuzz.ratio(suff, ' '.join(gram).lower())
            if score == 100:
                argmax = j
                break
            scores.append(score)
        else:
            argmax = np.argmax(scores)
            if max(scores) < 80:
                warnings.warn(
                    'Max score < 80. Your rows may have gotten misaligned '
                    f'at row {i}: {chunk.text}'
                )
        punct_len = skip_1st + argmax + 2
        rows.append(' '.join(words[start_i:start_i + punct_len]))
        start_i += punct_len

    new_df = pd.DataFrame(rows, columns=['text'])
    return pd.concat((new_df, df.reset_index()[['start', 'duration']].copy()),
                     axis=1)


def na_index_chunks(chunk, mode='isnull', flat=False, col='text') -> list:
    """Given a chunk of a df that may contain null text rows, return a
    list of lists where each nested list contains the indices of a
    contiguous chunk of null rows.

    Parameters
    ----------
    chunk: pd.DataFrame
        DF that may have 1 or more rows where `col` is null.
    mode: str
        One of ('isnull', 'notnull') and determines which rows to retrieve.
    flat: bool
        If True, return a list of int indices. If False, return a list of lists
        where each nested list contains int indices for 1 contiguous chunk of
        non-null rows.
    col: str
        Name of column in `chunk` to check for null-ness.

    Returns
    -------
    list: See `flat` parameter documentation.
    """
    # Depending on mode, these are either nans or not nans.
    nans = chunk[getattr(chunk.text, mode)()]
    if nans.empty: return []
    last_idx = nans.index[-1]
    res = []
    curr_chunk = []
    prev = None
    for idx in nans.index:
        if prev is None or (idx == prev + 1):
            curr_chunk.append(idx)
        else:
            res.append(curr_chunk)
            curr_chunk = [idx]
        if idx == last_idx:
            res.append(curr_chunk)
        prev = idx
    return flatten(res) if flat else res


class UnpunctuatedTranscript:
    """An auto-generated transcript (probably from a youtube video). This
    provides some functionality to get punctuated or unpunctuated chunks.
    """

    def __init__(self, df_gen, **kwargs):
        """
        # TODO: docs for lots of methods

        Parameters
        ----------
        df_gen
        kwargs
        """
        self.df_gen = df_gen
        self.df_punct = self.df_gen.copy()
        self.df_punct['text'] = np.nan

        # Allow kwargs but no extra_kwargs at this point since the latter is
        # really meant to be a stopgap solution (which you can pass in when
        # calling the _punctuate_chunk method).
        self.manager = PromptManager('punctuate', verbose=False)
        self.kwargs = dict(kwargs)

    @property
    def df(self):
        return self.df_gen

    def _punctuate_chunk(self, df_chunk, extra_kwargs=None, **kwargs) -> str:
        # Don't use stream=True or return_full=True here. Just want a string.
        text = ' '.join(df_chunk.text)
        max_tokens = int(len(text.split()) * 2)
        kwargs = dict(self.kwargs, **kwargs, max_tokens=max_tokens)
        if kwargs.get('mock', False):
            kwargs['mock_func'] = punctuate_mock_func
        return self.manager.query(task='punctuate',
                                  text=text,
                                  extra_kwargs=extra_kwargs,
                                  **kwargs)[1]

    # This version only punctuates rows of the relevant chunk that haven't
    # been previously punctuated. While this is faster and cheaper, I've seen
    # some hints that the punctuation task may work better when we pass it
    # long-ish chunks of text and not little partial snippets. So it might
    # actually be better not to do this? I was going to say it might do better
    # if given full sentences rather than fragments, but I guess we can't
    # easily extract those without doing the actual punctuation step.
    def punctuated_chunk(self, start_idx, end_idx, punctuate,
                         align_kwargs=None, extra_kwargs=None,
                         **query_kwargs):
        unpunct_chunk = self.df_gen.loc[start_idx:end_idx, :]
        if not punctuate:
            return unpunct_chunk

        # If our punctuated df has any null chunks, we only want to punctuate
        # them if specifically asked to.
        chunk = self.df_punct.loc[start_idx:end_idx, :]
        if punctuate == 'if_cached' and chunk.text.isnull().sum() > 0:
            return unpunct_chunk

        # When the whole chunk is pre-punctuated, na_index_chunks returns
        # an empty list so we're not doing any unnecessary gpt3 querying.
        for idx in na_index_chunks(chunk):
            df_chunk = self.df_gen.loc[idx]
            text_punct = self._punctuate_chunk(df_chunk,
                                               extra_kwargs=extra_kwargs,
                                               **query_kwargs)
            df_chunk_punct = realign_punctuated_text(
                df_chunk, text_punct, **ifnone(align_kwargs, {})
            )
            # Use 'values' attribute because realignment func resets index.
            self.df_punct.loc[idx, 'text'] = df_chunk_punct.text.values
        return self.df_punct.loc[start_idx:end_idx, :]

    def clear_punctuations(self):
        self.df_punct['text'] = np.nan


class PunctuatedTranscript:
    """Transcript that was manually generated by a human (i.e. it has things
    like punctuation, capitalization, and is generally high quality). Like
    UnpunctuatedTranscript, this will usually come from YouTube (some popular
    videos have been transcribed by creators or fans).
    """

    # Automated YouTube transcripts seem to skip all punctuation except single
    # quotes. Just delete most but replace a few with a space. This risks
    # creating multiple spaces so we also try to replace those. Save space as
    # a variable to avoid typos and make it explicit when we want to replace
    # with a space vs. nothing.
    space = ' '
    punct_rm = '|'.join(re.escape(char) for char in punctuation
                        if char not in ('/', '-'))
    punct_space = '|'.join(map(re.escape, ['/', '-', space * 2, space * 3]))

    def __init__(self, df_gen, df_punct, **kwargs):
        """
        # TODO: many methods to document
        kwargs:
            Just for compatibility with UnpunctuatedTranscript, which needs
            these to specify args like 'rstrip' when loading a prompt.
        """
        self.df_gen = df_gen
        self.df_punct = df_punct

    @property
    def df(self):
        return self.df_punct

    def punctuated_chunk(self, start_idx, end_idx, punctuate=True, **kwargs):
        chunk = self.df_punct.loc[start_idx:end_idx, :]
        # Notice this covers case where punctuate is True or a string.
        if punctuate:
            return chunk
        return chunk.assign(
            text=lambda x: x.text.str.lower()
                .str.replace(self.punct_rm, '')
                .str.replace(self.punct_space, self.space)
        )

    def clear_punctuations(self):
        warnings.warn('This is a manual transcript so there are no gpt3 '
                      'punctuations to clear.')


@auto_repr
class Transcript:

    def __init__(self, url, **kwargs):
        self.url = url
        self.id = video_id(url)
        self._transcript = self._fetch_transcripts(url, **kwargs)
        self.is_generated = isinstance(self._transcript,
                                       UnpunctuatedTranscript)
        # These are technically the start times for the first and last time
        # segments, which are slightly different from the video start and end
        # times.
        self.start_time, self.end_time = self.df.start.ends(1)

    def _time_range(self, start, end) -> pd.DataFrame:
        assert end > start, 'End time must be later than start time.'
        assert start >= 0 and end >= 0, 'Times must be non-negative.'

        df = self.df
        if start < self.start_time:
            start_idx = 0
        else:
            start_idx = df.loc[df.start <= start].index[-1]

        if end > df.start.iloc[-1]:
            end_idx = df.tail(1).index[0]
        else:
            end_idx = df.loc[df.start >= end].index[0]
        return df.iloc[start_idx:end_idx + 1]

    def time_range(self, start, end, punctuate='if_cached', align_kwargs=None,
                   extra_kwargs=None, **query_kwargs) -> pd.DataFrame:
        chunk = self._time_range(start, end)
        return self._transcript.punctuated_chunk(*chunk.ends(1).index,
                                                 punctuate=punctuate,
                                                 align_kwargs=align_kwargs,
                                                 extra_kwargs=extra_kwargs,
                                                 **query_kwargs)

    def time_range_str(self, start, end, punctuate='if_cached',
                       full_sentences=True, max_trim=120, margin=3,
                       align_kwargs=None, extra_kwargs=None,
                       **query_kwargs) -> str:
        if full_sentences and self.is_generated and punctuate is False:
            warnings.warn('This is an autogenerated transcript, so calling '
                          'time_range_str() with full_sentences=True will '
                          'not work as expected when punctuate=False. We '
                          'suggest setting these to both be True or both be '
                          'False.')

        # Values outside the acceptable range are handled later anyway. Widen
        # our candidate window so we can trim off partial sentences. May need
        # to experiment with right adjustment size.
        if full_sentences:
            start = max(0, start - margin)
            end += margin

        rows = self.time_range(start, end, punctuate=punctuate,
                               align_kwargs=align_kwargs,
                               extra_kwargs=extra_kwargs, **query_kwargs)
        text = ' '.join(rows.text.values)
        if not full_sentences:
            return text
        return self._full_sentences(text, max_trim=max_trim)

    @staticmethod
    def _full_sentences(text, max_trim=120, chars=('.', '!', '?')) -> str:
        first_upper = re.search('[A-Z]', text)
        start_idx = 0 if first_upper is None else first_upper.start()
        if start_idx > max_trim: start_idx = 0
        # Rfind returns -1 for missing chars.
        end_idx = max(text.rfind(char) for char in chars)
        if end_idx == -1 or end_idx < len(text) - max_trim - 1:
            end_idx = None
        else:
            end_idx += 1
        return text[start_idx:end_idx]

    @property
    def df(self):
        return self._transcript.df

    def _fetch_transcripts(self, url, **kwargs):
        """Wrapper to fetch youtube transcripts and create the appropriate
        transcript object depending on whether a manually generated (i.e.
        punctuated) transcript was retrieved.

        Parameters
        ----------
        url: str
        verbose: bool
        """
        df_gen, df_man, _ = self.get_transcripts(
            url, verbose=kwargs.get('verbose', True)
        )
        if df_man is None:
            return UnpunctuatedTranscript(df_gen, **kwargs)
        else:
            return PunctuatedTranscript(df_gen, df_man, **kwargs)

    def punctuated_index(self, flat=True) -> list:
        """Get indices of rows which have already been punctuated."""
        return na_index_chunks(self._transcript.df_punct, 'notnull', flat)

    def unpunctuated_index(self, flat=True) -> list:
        """Get indices of rows which have not yet been punctuated."""
        return na_index_chunks(self._transcript.df_punct, 'isnull', flat)

    def punctuated_times(self):
        df = self.df
        res = []
        for chunk in self.punctuated_index(flat=False):
            end_row = df.loc[chunk[-1]]
            res.append((df.loc[chunk[0], 'start'],
                        end_row.start + end_row.duration))
        return res

    def punctuated_time_rows(self, chunk=False):
        idx = self.punctuated_index(flat=not chunk)
        # Don't use self.df, that points to unpunctuated version for generated
        # transcript.
        df = self._transcript.df_punct
        if chunk: return [df.loc[i] for i in idx]
        return df.loc[idx]

    def clear_punctuations(self):
        # Useful if we've been testing with mock calls and want to reset.
        self._transcript.clear_punctuations()

    @staticmethod
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
        and 'manual' (the latter two lap to pandas dfs or None if no
        transcript was found). Manual transcripts are human-created.
        Generated transcripts are a bit lower quality and tend to lack
        punctuation.
        """
        langs = ['en', 'en-GB']
        id_ = video_id(url)
        res = {'generated': None, 'manual': None}
        trans_list = YouTubeTranscriptApi.list_transcripts(id_)
        res['generated'] = trans_list.find_generated_transcript(langs)
        try:
            res['manual'] = trans_list.find_manually_created_transcript(langs)
        except NoTranscriptFound:
            if verbose: warnings.warn('No manual transcript found.')
        if verbose:
            non_eng = [k for k, v in res.items()
                       if v and ('United Kingdom' in v.language)]
            if non_eng:
                warnings.warn(
                    f'{non_eng} {"has" if len(non_eng) == 1 else "have"} '
                    'language en-GB, not en.'
                )
        return Args(**{k: pd.DataFrame(v.fetch()) if v else v
                       for k, v in res.items()},
                    id=id_)

    def __str__(self):
        return f'{type(self).__name__}(url={self.url}, ' \
               f'is_generated={self.is_generated})'