"""Utility functions for interacting with the gpt3 api."""

from collections.abc import Iterable, Mapping
import numpy as np
import openai
import os
from pathlib import Path
import sys
import warnings

from htools import load, select, bound_args, spacer
from jabberwocky.config import C
from jabberwocky.utils import strip, bold, load_yaml


def load_openai_api_key():
    """Load openai API key. This must either be an environment variable called
    OPENAI_API_KEY or placed in a text file at ~/.openai.

    Returns
    -------
    str
    """
    key = os.getenv('OPENAI_API_KEY')
    if not key:
        with open(Path('~/.openai').expanduser(), 'r') as f:
            key = f.read().strip()
    return key


def openai_auth():
    """Load openai api key and try to set it in the openai library. This must
    be done after importing openai.
    """
    os.environ['OPENAI_API_KEY'] = key = load_openai_api_key()
    try:
        module = sys.modules['openai']
        module.api_key = key
    except Exception as e:
        warnings.warn('openai library has not been imported. API key not set.')


def query_gpt3(prompt, engine_i=0, temperature=0.7, max_tokens=50,
               logprobs=None, stream=False, mock=False, return_full=False,
               strip_output=True, mock_func=None,
               **kwargs):
    """Convenience function to query gpt3.

    Parameters
    ----------
    prompt: str
    engine_i: int
        Corresponds to engines defined in config, where 0 is the cheapest, 3
        is the most expensive, etc.
    temperature: float
        Between 0 and 1. 0-0.4 is good for straightforward informational
        queries (e.g. reformatting, writing business emails) while 0.7-1 is
        good for more creative works.
    max_tokens: int
        Sets max response length. One token is ~.75 words.
    logprobs: int or None
        Get log probabilities for top n candidates at each time step.
    stream: bool
        If True, return an iterator instead of a str/tuple. See the returns
        section as the output is slightly different. I believe each chunk
        returns one token when stream is True.
    mock: bool
        If True, return a saved sample response instead of hitting the API
        in order to save tokens. Note that your other gpt3 kwargs
        (max_tokens, logprobs, kwargs) will be ignored. return_full will be
        respected since it affects the number of items returned - it's not a
        kwarg passed to the actual query function. Text is surrounded by
        <MOCK></MOCK> tags to make it obvious when mock is True (it's easy to
        forget to change the value of mock when switching back and forth).
    return_full: bool
        If True, return a third item which is the full response object.
        Otherwise we just return the prompt and response text.
    strip_output: bool
        If True, strip text returned by gpt3. Without this, many prompts have a
        leading space and/or trailing newlines due to the way examples are
        formatted.
    mock_func: None or function
        When mock=True, you can provide a function here that accepts the prompt
        and returns something which will be used as the mock text. Sample use
        case: when punctuating a transcript, the text realignment process may
        raise an error when loading a saved mock response. Therefore, we may
        want to write a mock_func that extracts the new input portion of the
        prompt (discarding instructions and examples). This option is
        unavailable in stream mode.
    kwargs: any
        Additional kwargs to pass to gpt3.
        Ex: presence_penalty, frequency_penalty (both floats in [0, 1]).

    Returns
    -------
    tuple or iterator: When stream=False, we return a tuple where the first
    item is the prompt (str) and the second is the response text(str). If
    return_full is True, a third item consisting of the whole response object
    is returned as well. When stream=True, we return an iterator where each
    step contains a single token. This will either be the text response alone
    (str) or a tuple of (text, response) if return_full is True. Unlike in
    non-streaming mode, we don't return the prompt - that seems less
    appropriate for many time steps.
    """
    if mock:
        res = load(C.mock_stream_paths[stream])
        if mock_func:
            if stream:
                raise NotImplementedError('mock_func unavailable when '
                                          'stream=True.')
            res.choices[0].text = mock_func(prompt)
    else:
        res = openai.Completion.create(
            engine=C.engines[engine_i],
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            logprobs=logprobs,
            stream=stream,
            **kwargs
        )

    # Extract text and return. Zip maintains lazy evaluation.
    if stream:
        texts = (strip(chunk.choices[0].text, strip_output) for chunk in res)
        return zip(texts, res) if return_full else texts
    else:
        output = (prompt, strip(res.choices[0].text, strip_output), res)
        return output if return_full else output[:-1]


def query_content_filter(text):
    """Wrapper to determine if a piece of text is safe, sensitive, or unsafe.
    Details on these categories are here:
    https://beta.openai.com/docs/engines/content-filter
    This endpoint is free.

    Parameters
    ----------
    text: str

    Returns
    -------
    tuple[int, float]: First value is predicted class, where 0 is safe,
    1 is sensitive (politics/religion/race/suicide etc.), and 2 is unsafe
    (profane/prejudiced/otherwise NSFW).
    """
    res = openai.Completion.create(
        engine='content-filter-alpha-c4',
        prompt='<|end-of-text|>' + text + '\n--\nLabel:',
        temperature=0,
        max_tokens=1,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        logprobs=10
    )
    label = res.choices[0].text
    cls2logp = {x: res.choices[0].logprobs.top_logprobs[0]
        .get(x, float('-inf'))
                for x in ['0', '1', '2']}
    logp = cls2logp.pop(label)
    # If model is not confident in prediction of 2, choose the next most
    # likely class. See https://beta.openai.com/docs/engines/content-filter.
    if label == '2' and logp < -.355 and cls2logp:
        label, logp = max(cls2logp.items(), key=lambda x: x[-1])
    return int(label), np.exp(logp)


class PromptManager:
    """Simple class that stores all the prompt templates and default kwargs
    so we don't need to load them repeatedly. Use this as an interface for
    performing tasks on a video Transcript object.
    """

    def __init__(self, *tasks, verbose=True):
        """
        Parameters
        ----------
        prompts: str
            Optional way to provide one or more task names (these must be the
            names of existing subdirectories in the data/prompts directory).
            Example: PromptManager('tldr', 'punctuate')
            If none are provided, we load all available prompts in that
            directory.
        verbose: bool
            Passed to `load_prompt`. Might decide to use this elsewhere as well
            later.
        """
        self.verbose = verbose
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
            name2kwargs[path.stem] = load_prompt(path.stem,
                                                 verbose=self.verbose)
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


def print_response(prompt, response):
    """Print gpt3 prompt and response. Prompt is in bold to make it easier to
    distinguish between them.

    Parameters
    ----------
    prompt: str
    response: str
        The text returned by gpt3.
    """
    print(bold(prompt), end='')
    print(response)


def load_prompt(name, prompt='', rstrip=True, verbose=True):
    """Load a gpt3 prompt from data/prompts. Note that this function went
    through several iterations and early versions of this function didn't
    allow for an input prompt parameter. This worked fine for toy examples
    where the prompt is static (e.g. reformatting some dates) but as we get to
    more powerful prompts we often want to specify recommended hyperparameters
    and allow for inputting new text, so a yaml file became more appropriate.
    However, getting new lines, special characters, and brackets (for string
    formatting) to all work in yaml files turns out to be surprisingly hard, so
    we instead place the prompt in its own .txt file and leave the .yaml file
    for hypers.

    Parameters
    ----------
    name: str
        Name of subdirectory in data/prompts. Ex: 'simplify_ml'
    prompt: str
        Additional input to be inserted into the prompt template. For example,
        our tldr template prompt is "{}\n\ntl;dr:". We need to pass in text
        to summarize (this replaces the brackets like in a python f-string).
    rstrip: bool
        This is a safety measure to prevent us from accidentally leaving a
        trailing space after the end of the prompt (which leads to worse gpt3
        completions). We let the user turn it off in case a prompt requires it.
    verbose: bool
        If True, this will print a message on loading if one is specified in
        the prompt config file. This can be some kind of reminder or usage
        note.

    Returns
    -------
    dict: Keys are all kwargs for query_gpt3(). You may want to override some
    of these at times, but they at least provide reasonable defaults. Some are
    more important than others: for example, a 'stop' value will likely always
    be relevant, while 'max_tokens' or 'engine_i' may depend on the specific
    usage.
    """
    dir_ = Path(f'data/prompts/{name}')
    prompt_fmt = load(dir_/'prompt.txt')
    kwargs = load_yaml(dir_/'config.yaml')
    # If no prompt is passed in, we load the template and store it for later.
    if prompt:
        prompt = prompt_fmt.format(prompt)
    else:
        prompt = prompt_fmt
    # Vim adds trailing newline, which can hurt gpt3 quality.
    if rstrip: prompt = prompt.rstrip()
    kwargs['prompt'] = prompt
    msg = kwargs.pop('reminder', None)
    if msg and verbose: print(f'{name}: {msg}{spacer()}')
    return kwargs


def punctuate_mock_func(prompt, random_punct=True, sentence_len=15):
    """
    #TODO docs

    Parameters
    ----------
    prompt
    random_punct
    sentence_len

    Returns
    -------

    """
    text = prompt.rpartition('\n\nPassage: ')[-1]\
                 .rpartition('\n\nPassage with punctuation:')[0]
    if random_punct:
        words = text.split(' ')
        new_words = []
        for idx in range(0, max(sentence_len, len(words)), sentence_len):
            new_words.append(
                ' '.join(words[idx:idx+sentence_len]).capitalize() + '.'
            )
        text = ' '.join(new_words)
    return text


# I figure if we're importing these functions, we'll need to authenticate.
openai_auth()
