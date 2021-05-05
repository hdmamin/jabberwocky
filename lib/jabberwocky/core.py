from collections.abc import Iterable, Mapping
from fuzzywuzzy import fuzz
import numpy as np
import pandas as pd
from pathlib import Path
import re
import warnings

from htools import select, bound_args, spacer
from jabberwocky.openai_utils import query_gpt3
from jabberwocky.utils import load_prompt


class PromptManager:
    """Simple class that stores all the prompt templates and default kwargs
    so we don't need to load them repeatedly. Use this as an interface for
    performing tasks on a video Transcript object.
    """

    def __init__(self, *tasks):
        """
        Parameters
        ----------
        prompts: str
            Optional way to provide one or more task names (these must be the
            names of existing subdirectories in the data/prompts directory).
            Example: PromptManager('tldr', 'punctuate')
            If none are provided, we load all available prompts in that
            directory.
        """
        self.prompts = self._load_templates(set(tasks))

    def _load_templates(self, tasks):
        """Load template and default hyperparameters for each prompt.

        Parameters
        ----------
        tasks: Iterable[str]
            If empty, we load all available prompts in the data/prompts
            directory.

        Returns
        -------
        dict[str, dict]: Maps task name to dict of hyperparameters
        (including the prompt template).
        """
        name2kwargs = {}
        dir_ = Path('data/prompts')
        paths = (dir_ / p for p in tasks) if tasks else dir_.iterdir()
        for path in paths:
            if not path.is_dir():
                if tasks: warnings.warn(f'{path} is not a directory.')
                continue
            name2kwargs[path.stem] = load_prompt(path.stem)
        return name2kwargs

    def query(self, task, text, debug=False, extra_kwargs=None, **kwargs):
        """Query gpt3.

        Parameters
        ----------
        task: str
            Name of task, e.g. "tldr".
        text: str
            New text to insert into prompt template. Some toy prompts
            (e.g. "short_dates") don't need this and merely accept an empty
            string.
        debug: bool
            If True, don't actually query gpt3 - just print kwargs. This can be
            useful to make sure everything resolves to what we expect since I
            ultimately decided to stop pursuing my hacky dynamic method
            generation approach.
        extra_kwargs: dict or None
            If provided, these kwargs are used to update, NOT OVERWRITE,
            existing defaults. The typical use case would be if we want to add
            1 extra "stop" phrase without erasing the others. The value must be
            a list or dict: primitive kwargs can only be overwritten since it's
            not clear what the desired behavior would be otherwise.
        kwargs: any
            Overwrite default query_gpt3 kwargs for this single call. For
            example, maybe our 'tldr' task usually uses engine_i=3, but we want
            to test it with engine_i=0 once. We can pass in that value here,
            but future method calls without that kwarg will use the expected
            default behavior of engine_i=3.

        Returns
        -------
        tuple or iterator: Output of query_gpt3(). By default, a tuple of
        (prompt, response), but we can alter this by setting return_full=True
        (which adds the whole api response as a third item) or stream=True
        (which returns a generator instead of a tuple).
        """
        kwargs = self.kwargs(task=task, fully_resolved=False,
                             return_prompt=True, extra_kwargs=extra_kwargs,
                             **kwargs)
        prompt = kwargs.pop('prompt').format(text)
        if debug:
            print('prompt:\n' + prompt)
            print(spacer())
            print('kwargs:\n', kwargs)
            print(spacer())
            print('fully resolved kwargs:\n',
                  dict(bound_args(query_gpt3, [], kwargs)))
            return
        return query_gpt3(prompt, **kwargs)

    def kwargs(self, task, fully_resolved=True, return_prompt=False,
               extra_kwargs=None, **kwargs):
        """A nice way to sanity-check your gpt3 query hyperparameters to make
        sure everything resolves as expected (reconciling defaults stored by
        the class, function defaults, and potentially new kwargs passed in for
        a single method call. This is similar to calling query() in debug mode,
        though the latter also prints the prompt itself while this focuses
        solely on hyperparameters.

        Parameters
        ----------
        task: str
            Name of task, e.g. "tldr".
        fully_resolved: bool
            If True, combine new kwargs (if present) with prompt's defaults
            (stored by class) and query_gpt3() function defaults. Otherwise,
            we only resolve new kwargs with the prompt's defaults but not with
            query_gpt3().
        return_prompt: bool
            If False, remove the prompt from the returned dict. It's often long
            and when calling this method directly, we tend to be more concerned
            with the hyperparameters - if we want to see the prompt too, we
            can call query() in debug mode.
        extra_kwargs: dict
            Kwargs to update defaults with rather than overwriting them (as
            kwargs do). Values must be lists or dicts. Example: pass an extra
            string to use as a 'stop' parameter without erasing your default
            stop parameters.
        kwargs: any
            Parameters which will overwrite default hyperparameters.

        Returns
        -------
        dict
        """
        kwargs = {**self.prompts[task], **kwargs}
        for k, v in (extra_kwargs or {}).items():
            v_cls = type(v)
            # Make a new object instead of just using get() or setdefault
            # since the latter two methods both mutate our default kwargs.
            curr_val = v_cls(kwargs.get(k, v_cls()))
            if isinstance(v, Iterable):
                curr_val.extend(v)
            elif isinstance(v, Mapping):
                curr_val.update(v)
            else:
                raise TypeError(f'Key {k} has unrecognized type {v_cls} in '
                                '`extra_kwargs`.')
            kwargs[k] = curr_val

        if fully_resolved: kwargs = dict(bound_args(query_gpt3, [], kwargs))
        return kwargs if return_prompt else select(kwargs, drop=['prompt'])

    def prompt(self, task, text='', print_=False):
        """Another way to see what a prompt likes for a given task, with or
        without a piece of input text.

        Parameters
        ----------
        task: str
            One of the task names in self.
        text: str
            If provided, this will be inserted into the prompt template in the
            usual manner. Otherwise, we just see the unchanged template.
        print_: bool
            If True, print output instead of returning. This lets us see
            formatting such as newlines more easily.

        Returns
        -------
        str or None: Return None if print_ is True.
        """
        template = self.prompts[task]['prompt']
        res = template.format(text) if text else template
        if print_:
            print(res)
        else:
            return res

    def __repr__(self):
        return f'{type(self).__name__}({", ".join(map(repr, self.prompts))})'

    def __iter__(self):
        """Iterates over prompt names (strings)."""
        return iter(self.prompts)


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
