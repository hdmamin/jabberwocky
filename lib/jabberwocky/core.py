"""Core functionality that ties together multiple APIs."""

from fuzzywuzzy import fuzz
import numpy as np
import pandas as pd
import re
import warnings


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
