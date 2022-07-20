"""Utility functions for interacting with the gpt3 api.

A note on query functions:
There are a number of different services (both paid and free) that provide
access to GPT-like models. GPTBackend.query() provides a convenient interface
to them by calling different query functions under the hood. These query
functions are all defined in this module and have names starting with
'query_gpt'. There are a number of steps to defining a new query function:

1. Use the `mark` decorator to set "batch_support" to either True or False.
True means you can pass in a list of prompts, False means the prompt must be a
single string. GPTBackend will support batching either way, but it needs to
know whether each query function supports this natively in order to determine
how to do this.
2. The first parameter should be 'prompt' with no default. The other parameters
should have defaults, but keep in mind many of these will be ignored when
called by GPTBackend.query - the latter has its own defaults which are passed
in as kwargs. Speaking of which...
3. It must accept kwargs.
4. If it supports multiple completions, include a parameter "n" with a default
value of 1 in the signature. GPTBackend.query will support it either way, but
again it needs to know how to achieve this.
5. If multiple engines are available through this backend, update
jabberwocky.config.C.backend_engines, using the other examples as guidance. In
addition, in this scenario we recommend retrieving the engine name explicitly
as query_gpt_huggingface does:
```engine = GPTBackend.engine(engine, backend='huggingface')```
rather than simply calling ```GPTBackend.engine(engine)```.
6. We recommend popping off kwargs that you actually do want to use and then
providing a warning so the user can see the remaining unused kwargs if there
are any.
7. The fuction should either return a (str, dict-like) tuple or
(list[str], list[dict-like]). Use the latter if batch_support is True and
multiple completions per prompt are supported (i.e. n is in its parameters).
Technically I suppose there could be a backend that supported 1 but not the
other, but I haven't seen it yet so I'll figure that case out if/when needed.
8. When done, update some or all of the following GPTBackend class attributes.
- name2base
- name2func
- skip_trunc
Use the comments and examples in the class as guidance.
"""

import banana_dev as banana
from collections.abc import Iterable, Mapping
from collections import Counter, defaultdict, deque
from contextlib import contextmanager
from datetime import datetime
from fuzzywuzzy import process
from itertools import zip_longest, chain
import json
from nltk.tokenize import sent_tokenize
import numpy as np
import openai
import os
import pandas as pd
from pathlib import Path
import requests
import shutil
import sys
from threading import Lock, Thread
import time
import warnings
import yaml

from htools import load, select, bound_args, spacer, valuecheck, tolist, save,\
    listlike, Results, flatten, add_docstring, func_name, params, mark, \
    random_str, deprecated, xor_none, MultiLogger, eprint
from jabberwocky.config import C
from jabberwocky.external_data import wiki_data
from jabberwocky.streaming import stream_response, truncate_at_first_stop
from jabberwocky.utils import strip, bold, load_yaml, colored, \
    hooked_generator, load_api_key, with_signature, JsonlinesLogger,\
    thread_starmap, ReturningThread, containerize, touch, save_yaml, \
    seconds_til_midnight


HF_API_KEY = load_api_key('huggingface', raise_error=False)
BANANA_API_KEY = load_api_key('banana', raise_error=False)

# Ex: MOCKS[True, True, False] means
# n_prompts > 1, n_completions > 1, stream=False. N > 1 always means 2 in this
# case.
try:
    MOCKS = load(C.all_mocks_path)
except FileNotFoundError:
    warnings.warn(f'File {C.all_mocks_path} not found. The "mock" backend '
                  f'will be unavailable.')
    MOCKS = {}


@mark(batch_support=False)
def query_gpt_j(prompt, temperature=0.7, max_tokens=50, top_p=1.0, **kwargs):
    """Queries free GPT-J API. GPT-J has 6 billion parameters and is, roughly
    speaking, the open-source equivalent of Curie (3/19/22 update: size sounds
    more like Babbage actually). It was trained on more
    code than GPT3 though so it may do surprisingly well at those kinds of
    tasks.

    API uptime may be questionable though. There's an accompanying front end
    here:
    http://api.vicgalle.net:8000/

    Parameters
    ----------
    prompt: str
    temperature: float
    top_p: float
        Value in (0.0, 1.0] that determines how surprising the generation is.
        Lower values are more predictable (closer to argmax sampling). Often
        see the recommendation to change either this OR temperature, not both.
    max_tokens: int
    kwargs: any
        Only supported options are top_p (float) and stop (Iterable[str]).
        Notice that stream mode is not supported.

    Returns
    -------
    tuple[str, dict]: Response text, full response dict.
    """
    params = {'context': prompt,
              'token_max_length': max_tokens,
              'temperature': temperature,
              'top_p': top_p}

    # Ensure that we end up with a list AND that stop is still Falsy if user
    # explicitly passes in stop=None.
    stop = tolist(kwargs.pop('stop', None) or [])
    if stop: params['stop_sequence'] = stop[0]

    # Must keep this after the block of stop-related logic above.
    if kwargs:
        warnings.warn(f'GPT-J api does not support other kwargs: {kwargs}')

    res = requests.post('http://api.vicgalle.net:5000/generate',
                        params=params)
    res.raise_for_status()
    res = res.json()
    # Structure: text, full response
    # str, dict
    return res['text'], res


@mark(batch_support=True)
def query_gpt_huggingface(
        prompt, model=0, temperature=1.0, repetition_penalty=None,
        max_tokens=50, top_k=None, top_p=None, n=1, **kwargs
):
    """Query EleuetherAI gpt models using the Huggingface API. This was called
    query_gpt_neo in a former version of the library (which is used by the
    GUI) but the API now hosts a GPT-J model as well so I renamed it.

    Parameters
    ----------
    prompt: str
    model: int or str
        Determines which Huggingface model API to query. See
        config.C.backend_engines['huggingface'].
        Those names refer to the number of
        parameters in the model, where bigger models generally produce higher
        quality results but may be slower (in addition to the actual inference
        being slower to produce, the better models are also more popular so the
        API is hit with more requests).
    temperature: float
        Between 0 and 1. 0-0.4 is good for straightforward informational
        queries (e.g. reformatting, writing business emails) while 0.7-1 is
        good for more creative works. Warning: huggingface docs say this
        actually goes from 0-100 - should check if they're using this value
        differently than the openai API.
    top_k: None or int
        Kind of like top_p in that smaller values may produce more
        sensible but less creative responses. While top_p limits options to
        a cumulative percentage, top_k limits it to a discrete number of
        top choices.
    top_p: None or float
        Value in [0.0, 1.0] if provided. Kind of like temperature in that
        smaller values may produce more sensible but less creative responses.
    repetition_penalty
    max_tokens: int
        Sets max response length. One token is ~.75 words.
    n: int
        Number of completions to generate.
    kwargs: any
        Just lets us absorb extra kwargs when used in place of query_gpt3().

    Returns
    -------
    tuple[List[str], List[dict]]: First item contains string completions,
    second contains dicts with some additional metadata.  Index i in each list
    gives us the ith completion.
    """
    # Hardcode backend in case we use this function outside of the
    # GPTBackend.query wrapper.
    engine = GPT.engine(model, backend='huggingface')
    if engine is None:
        raise ValueError('Could not resolve engine for huggingface backend.')

    # Docs say we can return up to 256 tokens but API sometimes throws errors
    # if we go above 250.
    headers = {'Authorization': f'Bearer api_{HF_API_KEY}'}
    # Notice the names don't always align with parameter names - I wanted
    # those to be more consistent with query_gpt3() function. Also notice
    # that types matter: if Huggingface expects a float but gets an int, we'll
    # get an error.
    if repetition_penalty is not None:
        repetition_penalty = float(repetition_penalty)
    # Stopword truncation happens in GPT.query(). Remove this here so we don't
    # get warned about it being unused.
    kwargs.pop('stop', [])
    if kwargs:
        warnings.warn('query_gpt_huggingface received unused kwargs '
                      f'{kwargs}.')

    data = {'inputs': prompt,
            'parameters': {'top_k': top_k, 'top_p': top_p,
                           'temperature': float(temperature),
                           'max_new_tokens': min(max_tokens, 250),
                           'repetition_penalty': repetition_penalty,
                           'return_full_text': False,
                           'num_return_sequences': n}}
    url = f'https://api-inference.huggingface.co/models/EleutherAI/{engine}'
    r = requests.post(url, headers=headers, data=json.dumps(data))
    r.raise_for_status()
    res = r.json()
    # Structure: text, full response
    # List[str], List[dict]
    return [row['generated_text'] for row in res], res


def postprocess_gpt_response(response, stream=False):
    """Convert a raw gpt3 (openai/gooseai) response to the output returned by
    query_gpt3 (a list or generator of (text, dict) tuples). Keeping this as a
    separate function makes it easier to test out new functionality: we can
    pass a saved API response into this function to make sure everything works
    without calling a paid API every time.
    """
    # Extract text and return. Zip maintains lazy evaluation.
    if stream:
        # # Each item in zipped object is (str, dict-like).
        # texts = (chunk['choices'][0]['text'] for chunk in response)
        # chunks = (dict(chunk['choices'][0]) for chunk in response)
        # # Yields (str, dict) tuples.
        # return zip(texts, chunks)

        # Yields (str, dict) tuples. Initially tried to construct two
        # generators separately and zip them, but we end up skipping every
        # other token in each generator.
        # See https://www.pythonmorsels.com/iterator-zip/
        return ((chunk['choices'][0]['text'], dict(chunk['choices'][0]))
                for chunk in response)

    # Structure: (List[str], List[dict])
    return [row.text for row in response['choices']], \
           [dict(choice) for choice in response['choices']]


@mark(batch_support=True)
def query_gpt3(prompt, model=0, temperature=0.7, top_p=.99,
               frequency_penalty=0.0, presence_penalty=0.0, max_tokens=50,
               logprobs=None, n=1, stream=False, logit_bias=None, **kwargs):
    """Convenience function to query gpt3. Mostly serves 2 purposes:
    1. Build in some mocking functionality for cheaper/free testing.
    2. Explicitly add some parameters and descriptions to the function
    docstring, since openai.Completion.create does not include most kwargs.

    This function's signature and docstring are used to update
    GPTBackend.query's corresponding attributes, as well as the docstring of
    a few other backend functions. This setup is largely a holdover from the
    original design where queries for all backends were routed through
    query_gpt3 rather than GPTBackend.query, but it's also slightly more
    convenient to have the source of truth be a fucntion rather than a method.
    For instance, I don't think we could use a method's signature/docstring
    to decorate another method from the same class if that need came up at some
    point.

    Parameters
    ----------
    prompt: str
    model: int or str
        Corresponds to models defined in config, where 0 is the cheapest, 3
        is the most expensive, etc.
    temperature: float
        Between 0 and 1. 0-0.4 is good for straightforward informational
        queries (e.g. reformatting, writing business emails) while 0.7-1 is
        good for more creative works.
    top_p: float
        Value in (0.0, 1.0] that limits the model to sample from tokens making
        up the top_p percent combined. I.e. higher values allow for more
        creativity (like high temperature) and low values are closer to argmax
        sampling (like low temperature). API recommends setting a sub-maximal
        value for at most one of this and temperature, not both. NLCA book
        suggests using both though. I suggest mostly relying on temperature
        since there seem to be better heuristics about how to modulate this
        to impact outputs in desirable ways, while using top_p just enough to
        eliminate the most unlikely of completions.
    frequency_penalty: float
        Value in [-2.0, 2.0] where larger (more positive) values more heavily
        penalize words that have already occurred frequently in the text.
        Usually reasonable to keep this in [0, 1].
    presence_penalty: float
        Value in [-2.0, 2.0] where larger (more positive) values more heavily
        penalize words that have already occurred in the text. Usually
        reasonable to keep this in [0, 1].
    max_tokens: int
        Sets max response length. One token is ~.75 words.
    logprobs: int or None
        Get log probabilities for top n candidates at each time step. This
        will only be useful if you set return_full=True.
    n: int
        Number of possible completions to return. Careful: values > 1 can add
        up quickly w.r.t. cost.
    stream: bool
        If True, return an iterator instead of a str/tuple. See the returns
        section as the output is slightly different. I believe each chunk
        returns one token when stream is True.
    logit_bias: dict or None
        If provided, should map string(s) (NUMERIC INDEX of word tokens,
        not the tokens themselves) to ints between -100 and 100 (inclusive?).
        Values in (-1, 1) should be used to nudge the model, while larger
        values can effectively ban/compel the model to use certain words.
    kwargs: any
        Additional kwargs to pass to gpt3. Most common one is "stop", a list of
        up to 4 strings to truncate completions on.

    Returns
    -------
    tuple[list[str], List[dict]] or generator-like: When stream=False, we
    return a tuple where the first item is the response text(str) and the
    second item is the whole response object (technically the value
    corresponding to the "choices" key, but that contains pretty much
    everything of interest). When stream=True, we return a generator-like
    (technically I forget if this is a generator or iterator, but basically
    it lets us get results on the fly) where each step contains a single
    token's worth of results (still a text, dict-like tuple).
    """
    # Note: previously didn't allow streaming mode with n > 1 but I think this
    # should work now. To re-enable this constraint, uncomment the two lines
    # below.
    # if stream and n > 1:
    #     raise RuntimeError('Stream=True and n>1 not supported.')

    # Do not hardcode backend in GPTBackend.engine() call because we use this
    # function for both openai and gooseai. Rely on method to get the current
    # backend.
    res = openai.Completion.create(
        engine=GPT.engine(model),
        prompt=prompt,
        temperature=temperature,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        max_tokens=max_tokens,
        logprobs=logprobs,
        n=n,
        stream=stream,
        logit_bias=logit_bias or {},
        **kwargs
    )

    # Keep this step in a separate function so we can easily apply it to new
    # response objects when developing new functionality.
    return postprocess_gpt_response(res, stream=stream)


@mark(batch_support=False)
def query_gpt_repeat(prompt, upper=True, **kwargs):
    """Mock func that just returns the prompt as the response. By default,
    we uppercase the responses to make it more obvious when looking at outputs
    that the function did execute successfully.

    Returns
    -------
    tuple[str, dict]: First item just repeats the prompt back to us but
    uppercases it to distinguish it from the input. Second item is an empty
    dict to match the interface for all our query functions.
    """
    if kwargs:
        warnings.warn(f'Unused kwargs {kwargs} received by query_gpt_repeat.')
    # Structure: text, full response
    # str, dict
    return prompt.upper() if upper else prompt, {}


@mark(batch_support=True)
def query_gpt_mock(prompt, n=1, stream=False, **kwargs):
    """Return mocked openai/gooseai responses without actually hitting the API.
    We provide 8 different types of responses based on varying values of n,
    stream, and whether the prompt is a single string or list of strings.
    It is therefore only useful to vary these arguments when calling this
    function.

    Returns
    -------
    tuple[list[str], list[dict]] or generator[str, dict]
    """
    if not MOCKS:
        raise RuntimeError('The "mock" backend is unavailable at the moment '
                           'because we could not find the cached response '
                           'files on your system. They should live at '
                           f'{C.all_mocks_path}.')
    if n > 2:
        warnings.warn(f'query_gpt_mock only supports n=1 or n=2, not n={n}. '
                      'Because you passed in a value > 1, we default to n=2.')
        n = 2
    if kwargs.get('model', 0):
        warnings.warn(f'query_gpt_mock actually used model=0.')
    if kwargs.get('max_tokens', 3) != 3:
        warnings.warn(f'query_gpt_mock actually used max_tokens=3.')
    if kwargs.get('logprobs', 3) != 3:
        warnings.warn(f'query_gpt_mock actually used logprobs=3.')
    if kwargs:
        warnings.warn(f'query_gpt_mock received unused kwargs: {kwargs}')

    np_ = False
    if listlike(prompt) and len(prompt) > 1:
        np_ = True
    nc = n > 1
    resp = MOCKS[np_, nc, stream]
    return postprocess_gpt_response(resp, stream=stream)


@mark(batch_support=False)
@add_docstring(query_gpt3)
def query_gpt_banana(prompt, temperature=.8, max_tokens=50, top_p=.8,
                     top_k=False, **kwargs):
    """Free gptj access. Unclear which version of the model they provide -
    guessing 6B params? The query_gpt3 docstring is included below for
    convenience, but that doesn't mean all of its parameters are supported;
    only those visible in the signature are (all kwargs are ignored here). For
    example, this does not natively support stop phrases or n_prompts > 1 or
    n_completions > 1.

    Returns
    -------
    tuple[str, dict]
    """
    if kwargs:
        warnings.warn(f'query_gpt_banana received unused kwargs {kwargs}.')

    params = {
        'text': prompt,
        'length': max_tokens,
        'temperature': temperature,
        'topP': top_p,
        'topK': top_k,
    }
    res = banana.run(api_key=BANANA_API_KEY, model_key='gptj',
                     model_inputs=params)
    # str, dict
    return res['modelOutputs'][0]['output'], res


def drop_last_fragment(text, response=None, punct='.!?:")]\''):
    """Try to drop a trailing fragment from a gpt completion (sometimes the
    model gets cut off by max_tokens).

    Parameters
    ----------
    text: str
        GPT string response from our standard query() usage:
        text, response = GPT.query(...)
    response: None or dict
        GPT full response dict from our standard query() usage:
        text, response = GPT.query(...)
    punct: str
        Characters that mark the end of a sentence (we consider each character
        separately - these are not word tokens). We're pretty generous here -
        the goal isn't a perfect truncation (if there even is such a thing when
        a completion is cut off mid-sentence), we just want to outperform
        the "never truncate" and "always truncate" strategies.

    Returns
    -------
    str: Input text with the last fragment removed, if one exists. If the text
    contains <=1 full sentence, it will be kept automatically, with the
    justification that no response is worse than a low quality response.
    """
    if response and response.get('finish_reason', 'length') != 'length':
        return text
    sentences = sent_tokenize(text)
    if len(sentences) > 1 and not sentences[-1].endswith(tuple(punct)):
        sentences = sentences[:-1]
    return ' '.join(sentences)


class MockTokenizer:
    """Mock tokenizer that should only be used when trying to estimate the
    number of tokens in a piece of text. This can be used as a tokenizer in
    EngineMap.estimate_cost(). Slight advantages: it's 2-3 orders of
    magnitude faster than actually tokenizing text with huggingface's gpt2
    tokenizer, and it's very fast/simple to load (don't even have to
    instantiate it, just call MockTokenizer.tokenize()). That being said,
    the huggingface tokenizer is still quite fast.
    """

    def __init__(self, multiplier=1.33):
        self.multiplier = multiplier
        self.tokenize = self._instance_tokenize

    @classmethod
    def tokenize(cls, text, multiplier=1.33):
        """
        multiplier: float
            Specifies the estimated number of tokens per word. Default is based
            on openai docs' estimate of its tokenizer's ratio.

        Returns
        -------
        list[str]
        """
        return cls._tokenize(text, multiplier=multiplier)

    def _instance_tokenize(self, text, multiplier=None):
        """If we instantiate a MockTokenizer object, this method will be used
        as the tokenize method (see how constructor overwrites self.tokenize).
        This lets us use self.multiplier as a fallback while still allowing
        the class to provide the user with the option of using the classmethod
        version instead. (To be clear: the classmethod will not be available
        from an instantiated object. You get one or the other for each
        class/object).

        Returns
        -------
        list[str]
        """
        return self._tokenize(text, multiplier=multiplier or self.multiplier)

    @classmethod
    def _tokenize(cls, text, multiplier=None):
        text = text.split()
        n_new = int(round(len(text) * (multiplier - 1)))
        return text + ['']*n_new


class EngineMap:
    """Lets us specify engines more flexibly and obtain equivalents for
    different backends. E.g. for an openai engine=0 or engine='ada' or
    engine='text-ada-001', what gooseai engine does this map to?
    """

    bases = [
        'ada',
        'babbage',
        'curie',
        'davinci'
    ]
    backend_engines = C.backend_engines
    paid_backends = {'openai', 'gooseai'}

    # Unlike gooseai, openai charges for both prompt and generation tokens.
    # We convert their prices to dollars per token.
    openai_prices = {
        'ada': {'per': .0008 / 1_000},
        'babbage': {'per': .0012 / 1_000},
        'curie': {'per': .0060 / 1_000},
        'davinci': {'per': .0600 / 1_000},
    }

    # Gooseai base prices (in dollars) cover the input and the first 25
    # tokens of the output. `Per` prices are dollars per token.
    gooseai_prices = {
        'gpt-neo-20b': {'base': 0.002650, 'per': 0.000063},
        'fairseq-13b': {'base': 0.001250, 'per': 0.000036},
        'fairseq-6-7b': {'base': 0.000450, 'per': 0.000012},
        'gpt-j-6b': {'base': 0.000450, 'per': 0.000012},
        'gpt-neo-2-7b': {'base': 0.000300, 'per': 0.000008},
        'fairseq-2-7b': {'base': 0.000300, 'per': 0.000008},
        'gpt-neo-1-3b': {'base': 0.000110, 'per': 0.000003},
        'fairseq-1-3b': {'base': 0.000110, 'per': 0.000003},
        'gpt-neo-125m': {'base': 0.000035, 'per': 0.000001},
        'fairseq-125m': {'base': 0.000035, 'per': 0.000001},
    }

    @classmethod
    def get(cls, model, backend=None, infer=True, default=None,
            openai_passthrough=True, basify=False):
        """
        Parameters
        ----------
        model: int or str
            See class docstring for details.
        backend: str or None
            Name of backend to use, e.g. "openai", "gooseai", etc. If none is
            provided, the current backend will be used.
        infer: bool
            If true and the specified backend does not have an engine matching
            the desired engine, we check for progressively weaker engines
            trying to find something to return.
        default: any
            Value to return if we fail to resolve the engine under the current
            backend.
        openai_passthrough: bool
            If True and openai is the specified backend and engine is a str,
            we simply return the input (we do at least check that it contains
            one of the bases, i.e. that "ada" or "davinci" etc. is present in
            the name. But otherwise we trust you to ensure this is a real
            engine name.
            E.g. if True, 'code-ada-001' -> 'code-ada-001'
            and 'ada' -> 'ada'.
            If False, 'code-ada-001' -> 'text-ada-001'
            and 'ada' -> 'text-ada-001'.
        basify: bool
            If True and backend is openai, convert the output to one of the
            openai base engines, e.g. 'ada' instead of 'text-ada-001'.
            This will not be applied to your
            `default` value if that gets returned. openai_passthrough also
            takes priority, so basify will be ignored in that scenario.
            It is also ignored if the backend is anything other than 'openai'.

        Returns
        -------
        str or None: Technically, can be any value if you specify a different
        `default`. But typically it will be the name of the current/specified
        backend's equivalent engine. This is best explained by example (below).

        Examples
        --------
        >>> EngineMap.get(1, 'gooseai')
        'gpt-j-6b'

        >>> with gpt('huggingface'):
        >>>     print(EngineMap.get('ada'))
        'gpt-neo-2.7B'

        >>> EngineMap.get('text-davinci-002', 'huggingface', infer=True)
        # UserWarning: No engine=text-davinci-002 equivalent for backend
        # huggingface.Trying to auto-infer best option.
        'gpt-j-6B'
        """
        # Store this for potential error message later.
        user_engine = model
        if isinstance(model, int):
            if model not in range(4):
                raise ValueError(
                    f'Received invalid model value: {model}. If model is '
                    'specified as an integer, it must lie in [0, 3].'
                )
            engine_i = model
        else:
            base = cls.openai_base_engine(model)
            engine_i = cls.bases.index(base)

        backend = backend or GPT.current()
        if backend not in cls.backend_engines:
            return default

        backend_engines = cls.backend_engines[backend]
        if backend == 'openai' and openai_passthrough \
                and isinstance(user_engine, str):
            if basify:
                warnings.warn('Basify=True is ignored when '
                              'openai_passthrough=True.')
            if user_engine not in cls.bases + backend_engines:
                # We do still have some basic validation above that checks
                # that one of the openai bases is present in the name.
                warnings.warn(
                    f'Allowing model "{model}" to pass through because '
                    f'openai_passthrough=True. We trust you to make sure this '
                    f'is a valid model.'
                )
            return user_engine

        model = backend_engines[engine_i]
        if not model:
            msg = f'No model={user_engine} equivalent for backend {backend}.'
            if infer:
                warnings.warn(msg + 'Trying to auto-infer best option.')
                while engine_i > 0 and not model:
                    engine_i -= 1
                    model = backend_engines[engine_i]
            else:
                warnings.warn(
                    f'No matching model found. With backend {backend}, your '
                    f'options are {list(filter(None, backend_engines))}.'
                )
                return default

        if 'code' in model and backend != 'openai':
            warnings.warn(f'{backend} backend does not provide code-specific '
                          'models at the moment. We\'re returning the closest'
                          ' generic model.')
        if basify and backend == 'openai':
            return cls.openai_base_engine(model)
        return model

    @classmethod
    def openai_base_engine(cls, model: str):
        """Extract openai base engine name (e.g. 'ada') from a potentially
        longer string (e.g. 'text-ada-001'). If you pass in the short name, it
        should just return itself.

        Parameters
        ----------
        model: str
            E.g. 'code-babbage-001', 'text-ada-001', 'davinci'

        Returns
        -------
        str
        """
        matches = [chunk for chunk in model.split('-')
                   if chunk in cls.bases]
        if not matches:
            raise ValueError(f'Model "{model}" does not contain any of the '
                             f'recognized openai bases {cls.bases}.')
        if len(matches) > 1:
            raise ValueError(f'Model "{model}" contains multiple matches '
                             f'among the recognized openai bases '
                             f'{cls.bases}.')

        return matches[0]

    @classmethod
    def estimate_cost(cls, completion_length, prompt_length=None, prompt=None,
                      engines=(0, 1, 2, 3), tokenizer=None, return_full=True):
        """Estimate the cost of a query when using gooseai vs. openai. This
        requires you to know (or estimate) the completion length. The simplest
        way to do this is to use the max_tokens, but for some prompts
        estimating may be more appropriate (e.g. if you mostly use stop words
        as guardrails and leave max_tokens higher than you expect to use in
        order to avoid truncating completions mid-sentence).

        Note: as of 7/1/22, it seems that at least the openai estimates tend to
        be a bit lower than the actual charges. I think this it's mostly that
        the MockTokenizer formula is overly optimistic
        (it was provided by openai though 🤨). If you want more accurate
        estimates, passing in tokenizer=transformers.GPT2Tokenizer should help.

        Parameters
        ----------
        completion_length
        prompt_length
        prompt
        engines: Iterable[int or str]
            Engine numbers to consider when computing prices. This should work
            with strings too (e.g. ["ada", "text-davinci-002"]), though note
            that price computations will only be based on the openai base
            engine (i.e. "text-davinci-002" -> "davinci").
        tokenizer: None or transformers.GPT2Tokenizer
            If you don't know the number of tokens in the input, you must pass
            in a tokenizer so this method can tokenize the input and count.
            My vague recollection is that some of the open source models may
            use a slightly different tokenizer but I'm not sure - regardless,
            this should still give a decent estimate. An alternative is to
            pass in a mock tokenizer (e.g. MockTokenizer(), but anything with a
            tokenize method, would work. If you're okay with MockTokenizer's
            default multiplier, you don't even have to instantiate it since it
            provides a classmethod version of tokenize as well.)
            if you want something faster/simpler. I benchmarked a
            ~500-600 token prompt and it only took ~3-7ms with the gpt2
            tokenizer, so speed probably isn't a big concern (though the mock
            version is 2-3 orders of magnitude faster).

        Returns
        -------
        dict: Keys "backend" and "engine" are strings corresponding to the
        cheapest option of the specified engines. "cost" is a float
        specifying the cost of the query (in dollars) using the recommended
        backend and engine. If return_full=True, and additional key "full" is
        included, which maps to a df containing the cost in dollars for all
        specified backend/engine options.
        """
        # Pass in engines=None to get ALL possible engine prices.
        xor_none(prompt_length, prompt)
        xor_none(prompt_length, tokenizer)

        if engines:
            engines = tolist(engines)
            openai_names = [cls.get(engine, 'openai', openai_passthrough=False,
                                    basify=True)
                            for engine in engines]
            gooseai_names = [cls.get(engine, backend='gooseai')
                             for engine in engines]

        prompt_length = prompt_length or len(tokenizer.tokenize(prompt))
        openai_prices = cls.openai_prices
        gooseai_prices = cls.gooseai_prices
        if engines:
            openai_prices = select(openai_prices, keep=openai_names)
            gooseai_prices = select(gooseai_prices, keep=gooseai_names)
        gooseai_resolved = [
            ('gooseai', name,
             prices['base'] + prices['per'] * max(0, completion_length - 25))
            for name, prices in gooseai_prices.items()
        ]
        openai_resolved = [
            ('openai', name,
             prices['per'] * (prompt_length + completion_length))
            for name, prices in openai_prices.items()
        ]

        # Prices are returned in dollars.
        df = pd.DataFrame(
            gooseai_resolved + openai_resolved,
            columns=['backend', 'engine', 'cost']
        ).sort_values('cost', ascending=True).reset_index(drop=True)
        res = df.iloc[0].to_dict()
        if return_full: res['full'] = df
        return res


class GPTBackend:
    """
    Examples
    --------
    gpt = GPTBackend()

    # Default backend is openai.
    openai_res = gpt.query(**kwargs)

    with gpt('gooseai'):
        # Now we're using the gooseai backend.
        gooseai_res = gpt.query(**kwargs)

    # Now we're back to using openai.
    openai_res_2 = gpt.query(**kwargs)

    # Now we'll switch to gooseai and changes will persist since we're not
    # using a context manager.
    gpt.switch('gooseai')
    gooseai_res_2 = gpt.query(**kwargs)
    """

    lock = Lock()

    # Only include backends here that actually should change the
    # openai.api_base value, a.k.a. those that use the openai.Completion.create
    # method. I also treat this as a record of which backends are paid, though
    # it's plausible that may change eventually.
    # 6/26/22 update: EngineMap.paid_backends now tracks this explicitly. Might
    # want to consider how we can handle this to avoid the risk of these two
    # attributes getting out of sync.
    name2base = {
        'openai': 'https://api.openai.com/v1',
        'gooseai': 'https://api.goose.ai/v1',
    }

    # Order matters: keep openai first so name2key initialization works.
    # 4/13/22 update: Is the above comment still true? I don't think so but
    # leave it for now just in case.
    name2func = {
        'openai': query_gpt3,
        'gooseai': query_gpt3,
        'huggingface': query_gpt_huggingface,
        'hobby': query_gpt_j,
        'banana': query_gpt_banana,
        'repeat': query_gpt_repeat,
        'mock': query_gpt_mock
    }

    # Backends that require an api key.
    needs_api_key = {
        'openai',
        'gooseai',
        'huggingface',
        'banana'
    }

    # Names of backends that perform stop word truncation how we want (i.e.
    # allow us to specify stop phrases AND truncate before the phrase rather
    # than after, if we encounter one).
    skip_trunc = {'openai'}

    name2key = {}
    for name in name2func:
        if name in needs_api_key:
            name2key[name] = load_api_key(name, raise_error=False)
        else:
            name2key[name] = f'<{name.upper()} BACKEND: FAKE API KEY>'

    def __init__(self, date_fmt="%Y.%m.%d", log_stdout=True):
        self.new_name = ''
        self.old_name = ''
        self.old_key = ''
        self.verbose_switch = True
        self.old_verbose_switch = self.verbose_switch
        self.date_fmt = date_fmt
        self.logger = JsonlinesLogger(
            C.root/
            f'data/logs/{datetime.today().strftime(date_fmt)}.jsonlines'
        )
        # Even if we remove stdout handler, jsonlines logging will still occur.
        # Disabling the jsonlines logger can still be done at the query level
        # with log=False but it usually makes sense to keep it.
        if not log_stdout:
            self.disable_stdout_logging()
        self.thread = Thread(target=self.update_log_path, daemon=True).start()

    def __call__(self, name, verbose=True):
        """__enter__ can't take arguments so we need to specify this here.
        Notice that name is auto-lowercased and spaces are removed.
        """
        new_name = name.lower().replace(' ', '')
        if new_name not in self.name2func:
            raise ValueError(f'Invalid name {name}. Valid options are: '
                             f'{list(self.name2func)}')

        self.new_name = new_name
        self.old_name = self.current()
        self.old_verbose_switch, self.verbose_switch = self.verbose_switch, \
                                                       verbose
        return self

    def __enter__(self):
        """Change backend to the one specified in __call__, which is
        automatically called first when using `with` syntax.
        """
        if self.verbose_switch:
            print(f'Switching openai backend to "{self.new_name}".')
        # Store an attribute on openai itself to reduce risk of bugs caused by
        # GPTBackend being deleted or recreated. Previously used a
        # self.base2name mapping to retrieve the current name but that doesn't
        # work when multiple names use the same base (e.g. huggingface and
        # hobby API backends can't be identified just by their base with
        # this implementation).
        openai.curr_name = self.new_name
        self.old_key, openai.api_key = openai.api_key, \
                                       self.name2key[self.new_name]
        if self.new_name in self.name2base:
            openai.api_base = self.name2base[self.new_name]

    def __exit__(self, exc_type, exc_val, traceback):
        """Revert to previously used backend on contextmanager exit."""
        if self.verbose_switch:
            print(f'Switching  backend back to "{self.old_name}".')
        self.verbose_switch = self.old_verbose_switch
        openai.api_key = self.old_key
        if self.old_name in self.name2base:
            openai.api_base = self.name2base[self.old_name]
        openai.curr_name = self.old_name
        self.clear()

    @classmethod
    def ls(cls):
        """Print current state of the backend: api_base, api_key, and
        query_func. Mostly useful for debugging and sanity checks.
        """
        print('\nBase:', openai.api_base)
        print('Query func:', cls._get_query_func())

    @classmethod
    def backends(cls):
        """List all valid backend names. We could always access these via a
        class attribute but this name is easier to remember.

        Returns
        -------
        list[str]
        """
        return list(cls.name2func)

    def clear(self):
        """Reset instance variables tracking that were used to restore
        previous backend.
        """
        self.old_key = self.old_name = self.new_name = ''

    def switch(self, name):
        """Switch backend and make changes persist, unlike in context manager
        where we reset them on exit.

        Parameters
        ----------
        name: str
            One of (openai, gooseai).
        """
        self(name=name).__enter__()
        self.clear()

    @staticmethod
    def current():
        """Get current backend name, e.g. "gooseai". If we've ever switched
        backend with GPTBackend, openai.curr_name
        should exist. If not, the backend should be the default.

        Returns
        -------
        str
        """
        return getattr(openai, 'curr_name', 'openai')

    @classmethod
    def _get_query_func(cls, backend=None):
        """Return current mock function (callable or None)."""
        return cls.name2func[backend or cls.current()]

    @classmethod
    def key(cls):
        """Return current API key. In some cases this is a mock value since
        some modes don't have a key.
        """
        # More reliable than checking name2key because the openai attribute
        # is what's actually used (at least for openai vs. gooseai -
        # huggingface query_func technically uses a global).
        return openai.api_key

    @classmethod
    @add_docstring(EngineMap.get)
    def engine(cls, engine, backend=None, **kwargs):
        """Get appropriate engine name depending on current api backend.

        Parameters
        ----------
        engine: int or str
            If a str, this is an engine name like 'davinci' or
            'text-davinci-002' (both work, but depending on other args may
            behave differently). If an int, this is a number from 0-3
            (inclusive). Openai
            and gooseai *should* perform similar for values of 0-2, but
            openai's 3 (davinci, 175 billion parameters) is a much bigger model
            than gooseai's 3 (NEO-X, 20 billion parameters). Mostly used in
            query_gpt3().
        backend: str or None
            If provided, should be the name of a backend (e.g. 'huggingface'
            or any of the keys in GPTBackend.backends()).

        Returns
        -------
        str: Name of an engine, e.g. "text-davinci-002" if we're in openai mode
        or "gpt-neo-20b" if we're in gooseai mode.
        """
        return EngineMap.get(engine, backend=backend, **kwargs)

    @contextmanager
    def _optimized(self, prompt, optimize_cost=False, warning_sleep=3,
                   **kwargs):
        """Context manager to optionally change the backend to optimize cost.

        Parameters
        ----------
        optimize_cost
        warning_sleep

        Returns
        -------

        """
        try:
            if optimize_cost:
                if self.current() not in self.name2base:
                    warnings.warn(
                        'You\'re currently using a free backend '
                        f'({self.current()}), but you set optimize_cost=True '
                        'which will result in charges. Waiting '
                        f'{warning_sleep} seconds to give you time to '
                        'cancel your query...'
                    )
                    time.sleep(warning_sleep)

                # Use gpt3 func instead of query_func because we only care
                # about openai and gooseai when considering cost.
                defaults = {k: v.default for k, v in
                            params(query_gpt3).items()}
                c_len = kwargs.get('max_tokens', defaults['max_tokens'])
                cost_res = EngineMap.estimate_cost(
                    completion_length=c_len,
                    prompt=prompt,
                    engines=kwargs.get('model', defaults['model']),
                    tokenizer=MockTokenizer
                )
                print(cost_res.pop('full'))
                with self(cost_res['backend']):
                    yield
            else:
                yield
        finally:
            pass

    # Decorator order matters - doesn't work if we flip these. [5/1/22 update:
    # previous statement was true when this was a classmethod but not sure if
    # it's still true.
    # Keep=True in with_signature makes kwargs resolution work when passing in
    # kwargs not present in query_gpt3 (some backends may accept args that
    # openai backend doesn't. We could pass these in regardless, but
    # signature(query_func).bind_partial(**kwargs) only works with keep=True).
    @with_signature(query_gpt3, keep=True)
    @add_docstring(query_gpt3)
    def query(self, prompt, strip_output=True, log=True, optimize_cost=False,
              subwords=True, drop_fragment=False, **kwargs):
        """Query gpt3 with whatever the current backend is. Aside from prompt,
        you should pass in named arguments.

        Parameters
        ----------
        prompt: str or list[str]
        strip_output: bool
        log: bool or str
            If True, the logfile defaults to a path like
            './data/logs/2022.04.07.jsonlines' (current year, month, day).
            If str, use that as the log path. If False or None, do not log.
        subwords: bool
            If True and stream=True but the currentbackend doesn't natively
            support streaming, you have a choice of streaming words or
            subword tokens. In most cases subwords=True should be preferable,
            but if you're piping words to some kind of audio generation service
            it may be preferable to use words rather than subwords since
            subwords may not be pronounced correctly (even then, you'd probably
            want to pass in larger chunks of text to avoid a very choppy
            cadence).
        optimize_cost: bool
            If True, check if gooseai or openai is projected to be cheaper
            for the current engine (and based on the input prompt length and
            the maximum completion length) and automatically choose the cheaper
            option. This option expects you are already using one of the two
            paid backends: if you're using a free backend, it will warn you and
            pause for a few seconds to give you a chance to cancel before
            switching to a paid one.
        kwargs

        Returns
        -------
        list[str, dict] or generator[str, dict]
        """
        stream = kwargs.get('stream', False)
        if stream and strip_output:
            warnings.warn('Doesn\'t make sense to use strip_output=True in '
                          'streaming mode. Automatically setting it to False.')
            strip_output = False
        if stream and drop_fragment:
            warnings.warn('Drop_fragment=True is unavailable in streaming '
                          'mode and will be ignored.')
            drop_fragment = False

        stop = kwargs.get('stop', [])
        n_stop = len(tolist(stop))
        if n_stop > 4:
            warnings.warn('Parameter `stop` should container less than or '
                          f'equal to 4 strings. Found {stop}.')

        # V2 library no longer supports user passing in mock_func. We want to
        # remove this from the kwargs we pass to our actual function. This
        # condition shouldn't ever be triggered but I'm leaving it for now in
        # case there's some code I missed that still uses mock_func.
        if kwargs.pop('mock_func', None):
            warnings.warn(
                f'Encountered unexpected arg `mock_func`. This was part of '
                f'the v1 library but is no longer supported. The arg will be '
                f'ignored.'
            )

        with self._optimized(prompt, optimize_cost=optimize_cost,
                             **kwargs):
            query_func = self._get_query_func()
            if listlike(prompt) and not query_func.batch_support:
                query_meth = self._query_batch
            else:
                query_meth = self._query
            return query_meth(prompt, query_func=query_func,
                              strip_output=strip_output, log=log,
                              subwords=subwords, drop_fragment=drop_fragment,
                              **kwargs)

    def _query(self, prompt, query_func, strip_output=True, log=True,
               subwords=True, drop_fragment=False, **kwargs):
        """The base query method. We separate this from self.query because
        that may call self._query_batch which in turn calls this method. Before
        refactoring, this made for a rather confusing case where query could
        call _query_batch and _query_batch could also call query.
        """
        # Including indices in responses makes it easier to tell which
        # completions correspond to which prompts when we use multiple prompts
        # and/or multiple completions per prompt.
        start_i = kwargs.pop('start_i', 0)
        prompt_i = kwargs.pop('prompt_i', 0)
        n = kwargs.get('n', 1)
        np_ = len(prompt) if listlike(prompt) else 1
        kwargs['prompt'] = prompt
        self._log_query_kwargs(log=log, query_func=query_func,
                               version=kwargs.pop('version', None),
                               **kwargs)
        func_params = params(query_func)
        trunc_full = self.current() not in self.skip_trunc
        stream = kwargs.get('stream', False)
        stop = kwargs.get('stop', [])

        # Possibly easier for caller to check for errors this way? Mostly a
        # holdover from v1 library design, but I'm not 100% sure if the
        # benefits still hold given the new design.
        try:
            if n > 1 and 'n' not in func_params:
                del kwargs['n']
                # If current query function doesn't natively support multiple
                # completions, we can make multiple threaded requests. Need
                # to unzip afterwards to regain the (texts, full_responses)
                # structure.
                response = thread_starmap(query_func,
                                          [kwargs for _ in range(n)])
                response = list(zip(*response))
            else:
                response = query_func(**kwargs)
        except Exception as e:
            raise MockFunctionException(str(e)) from None

        if stream:
            return stream_response(response, start_i=start_i,
                                   prompt_i=prompt_i, n=n, np=np_,
                                   stop=stop, backend=self.current(),
                                   subwords=subwords)

        text, full_response = containerize(*response)
        # Manually check for stop phrases because most backends either don't
        # or truncate AFTER the stop phrase which is rarely what we want.
        clean_text = []
        clean_full = []
        for i, (text_, resp_) in enumerate(zip(text, full_response)):
            text_ = truncate_at_first_stop(
                text_,
                stop_phrases=stop,
                finish_reason=resp_.get('finish_reason', ''),
                trunc_full=trunc_full,
                trunc_partial=True
            )
            if drop_fragment: text_ = drop_last_fragment(text_, resp_)
            clean_text.append(strip(text_, strip_output))
            # Add prompt i so index is correct when calling this from
            # _query_batch. When not calling it from there, prompt_i is always
            # 0 so this does not affect the calculation.
            clean_full.append({**resp_, 'prompt_index': prompt_i + (i // n)})

        return clean_text, clean_full

    def _query_batch(self, prompts, query_func, **kwargs):
        """Get completions for k prompts in parallel using threads. This is
        only necessary for backends that don't natively support lists of
        prompts - both openai and gooseai provide similar functionality
        natively.

        Parameters
        ----------
        prompts: list[str]
        query_func: FunctionType

        Returns
        -------
        tuple[list] or generator: When stream is False (the default), we get
        a list of strings (completions) and a list of dicts (full responses),
        just like with query(). When stream is True, we get a generator (or
        something similar) of (text, full_response_dict) tuples. Note that
        prompts may be interleaved, i.e. we might get a token from prompt 1,
        then 1 from prompt 2, the prompt 1 again. For that reason, the full
        response dict is updated to include a prompt_index key-value pair,
        where prompt_index=0 means the completion corresponds to the first
        prompt you passed in.

        When stream=True, each full response also includes an "index" key-value
        pair. This is a completion index, which is useful because each tuple
        in this mode contains a single token so the response is generally
        only meaningful in context.
        """
        kwargs.update(query_func=query_func)
        # Setting start_i to i*n ensures that the 'index' returned in streamed
        # responses is different for each prompt's completion(s). Otherwise,
        # because each query is run separately, each prompt's completion(s)
        # would start at 0.
        n = kwargs.get('n', 1)
        threads = [
            ReturningThread(target=self._query, args=(prompt,),
                            kwargs={**kwargs, 'start_i': i * n, 'prompt_i': i})
            for i, prompt in enumerate(tolist(prompts))
        ]
        for thread in threads: thread.start()
        # Each item is a tuple of (list[str], list[dict]).
        res = [thread.join() for thread in threads]
        if kwargs.get('stream', False):
            return chain(*res)
        # Convert results to 2 lists of lists, then flatten each outer list.
        texts, fulls = map(list, zip(*res))
        return sum(texts, []), sum(fulls, [])

    def _log_query_kwargs(self, log, query_func=None, version=None, **kwargs):
        """Log kwargs for troubleshooting purposes."""
        if log:
            # Meta key is used to store any info we want to log but that should
            # not be passed to the actual query_gpt3 call.
            kwargs['meta'] = {
                'backend_name': self.current(),
                'query_func': func_name(query_func) if query_func else None,
                'datetime': datetime.now().ctime(),
                'version': version
            }
            with self.lock:
                if not isinstance(log, (str, Path)):
                    log = self.logger.path

                # If log file was deleted, we must recreate it AND use
                # change_path to reopen the file object.
                if not os.path.exists(log):
                    touch(log)
                    self.logger.path = None
                if log != self.logger.path:
                    self.logger.change_path(log)
            self.logger.info(kwargs)

    @classmethod
    def refresh_api_keys(cls):
        """Sometimes I change api keys in my local filesystem and re-importing
        the class doesn't seem to reload them. This does.
        """
        for name in cls.name2func:
            try:
                cls.name2key[name] = load_api_key(name)
            except FileNotFoundError:
                cls.name2key[name] = f'<{name.upper()} BACKEND: FAKE API KEY>'

    def disable_stdout_logging(self):
        # Stop the logger from printing to stdout. Unlike passing log=False to
        # the query method, this change persists. It does not affect jsonlines
        # logging.
        self.logger.remove_stdout_handler()

    def update_log_path(self):
        """Method that GPT runs in the background (threaded) to update the log
        filename nightly at midnight. This way we get one file per day instead
        of one enormous file total.
        """
        while True:
            dt = datetime.today()
            # First update may not be exactly at midnight but subsequent ones
            # should be. The exact switch time isn't particularly important.
            if dt.hour == 0:
                new_name = dt.strftime(self.date_fmt)
                with self.lock:
                    path = Path(self.logger.path)
                    *parts, _ = path.parts
                    self.logger.change_path(
                        os.path.join(*parts, f'{new_name}{path.suffix}')
                    )
            # Sleep til next midnight regardless of whether we made a change.
            time.sleep(seconds_til_midnight(dt))

    def __repr__(self):
        return f'{func_name(self)} <current_name: {self.current()}>'


class MockFunctionException(Exception):
    """Allow all mock query functions to return a common exception."""


class QueryAllowedResult:
    """Object returned by PriceMonitor.allowed().
    Truthy if the query should be allowed to proceed, falsy if api usage
    exceeded the allowed amount (specified when creating PriceMonitor).
    See __bool__ docstring for sample usage.
    """

    def __init__(self, error: bool, warn: bool, monitor):
        self.error = error
        self.warn = warn
        self.running_cost = monitor.running_cost
        self.time_window = monitor.time_window
        self.message = self._construct_message(monitor)

    def _construct_message(self, monitor):
        if self.error:
            adverb = 'extremely'
            limit = monitor.max_cost
        elif self.warn:
            adverb = 'suspiciously'
            limit = monitor.warn_cost
        else:
            return ''
        queue_str = '\n'.join(
            f"{dt.strftime('%Y/%m/%d %H:%M:%S')}: ${cost:.2f}"
            for dt, cost in monitor.q
        )
        return f'Your recent api usage looks {adverb} high. In the last ' \
               f'{self.time_window} seconds, you spent an estimated ' \
               f'${self.running_cost:.2f} (> ${limit:.2f}). Here is your ' \
               f'query queue:\n\n{queue_str}'

    def __bool__(self):
        """Returns True if no error was raised (i.e. we are allowed to
        proceed), False otherwise. If there is no error message but there is a
        warning, this will be falsy because we are allowed to proceed. This
        let us do something like:

        ```
        allowed = monitor.allowed(query)
        if not allowed:
            raise RuntimeError(allowed.message)
        elif allowed.warn:
            warnings.warn(allowed.message)
        GPT.query(query)
        ```
        """
        return not self.error


class PriceMonitor:

    def __init__(self, time_window=125, max_cost=0.6, warn_cost=0.15,
                 tokenizer=MockTokenizer):
        """
        time_window: int
            Specifies how long to look back when computing total price.
            Units are seconds. (One probably useless detail: for the default I
            added a few extra seconds instead of an exact multiple of minutes
            on the off chance this might flag a lazy bad actor who uses some
            fixed, human sensical wait time between queries.
        max_cost: float
            The amount (specified in dollars) where, if we spend at least this
            much in `time_window`, we should assume something fishy is going
            on). I seem to average around $0.03 every 2 minutes so the default
            value represents ~20x typical usage.
        warn_cost: float
            Similar to max_cost but a lower number. If total cost exceeds this
            within `time_window`, the user should be warned but the query
            should be allowed to proceed. The default value is ~5x typical
            usage.

        Examples
        --------
        # Allow at most $2 in the last 5 minutes.

        price_monitor = PriceMonitor(time_window=300, max_price=2.0)
        allowed = monitor.allowed(query, **query_kwargs)
        if not allowed:
            raise RuntimeError(allowed.message)
        elif allowed.warn:
            warnings.warn(allowed.message)
        GPT.query(query)
        """
        # This will store (datetime, price_in_dollars) tuples.
        self.q = deque()
        self.time_window = time_window
        self.max_cost = max_cost
        self.warn_cost = warn_cost or max_cost
        self.tokenizer = tokenizer
        self.running_cost = 0
        self.n_errors = 0
        if self.warn_cost > self.max_cost:
            raise ValueError('warn_cost must be <= max_cost.')

    def allowed(self, prompt, model, max_tokens, dt=None, backend=None,
                verbose=False):
        """Check if api usage exceeds the allowed amount.

        Parameters
        ----------
        prompt: str
            Fully resolved prompt that will be sent to api.
        model: int
            Value in [0, 1, 2, 3] corresponding to model like
            'ada', 'babbage', 'curie', 'davinci'.
        max_tokens: int
            Max length of the completion.
        dt: None or datetime.datetime
            Usually don't need to specify this - it defaults to the current
            time. Passing in a datetime object is mostly useful for testing.
        backend: None or str
            Usually don't need to specify this - it defaults to the current
            backend. Passing in a str is mostly useful for testing. In that
            case, it should be a string in ('gooseai', 'openai').
        verbose: bool
            If True, print the queue and running cost.

        Examples
        --------
        # Allow at most $2 in the last 5 minutes.

        price_monitor = PriceMonitor(time_window=300, max_price=2.0)
        allowed = monitor.allowed(query, **query_kwargs)
        if not allowed:
            raise RuntimeError(allowed.message)
        elif allowed.warn:
            warnings.warn(allowed.message)
        GPT.query(query)
        """
        dt = dt or datetime.now()
        backend = backend or GPT.current()
        if backend not in EngineMap.paid_backends:
            return QueryAllowedResult(error=False, warn=False, monitor=self)

        cost = EngineMap.estimate_cost(
            max_tokens, prompt=prompt, tokenizer=self.tokenizer,
            engines=[model], return_full=True
        )['full'].loc[lambda x: x.backend == backend, 'cost'].values[0]
        self.q.append((dt, cost))
        self.running_cost += cost
        while (dt - self.q[0][0]).total_seconds() > self.time_window:
            _, cur_cost = self.q.popleft()
            self.running_cost -= cur_cost
        if verbose:
            eprint([(dt.strftime('%H:%M:%S'), cost) for dt, cost in self.q])
            print('Running cost:', self.running_cost)
        error = self.running_cost >= self.max_cost
        if error: self.n_errors += 1
        warn = not error and (self.running_cost >= self.warn_cost)
        return QueryAllowedResult(error, warn, self)


GPT = GPTBackend()


def iter_engine_names(*backends, **kwargs):
    """Yields tuples of engine name strings for all user-specified backends.
    Mostly useful for testing purposes.

    Parameters
    ----------
    backends: str(s)
        E.g. 'gooseai', 'openai'. The order will correspond to the order of the
        names in each returned tuple.
    kwargs: any
        Additional kwargs passed on to GPTBackend.engine()
        (e.g. openai_passthrough, basify, etc.).

    Yields
    ------
    tuple[str]: Each tuple contains one string for each backend in the input
    backends arg (in the same order). There are a total of 4 tuples (yielded
    one at a time) corresponding to our 4 base engine levels.
    """
    for i in range(4):
        yield tuple(GPT.engine(i, backend=backend, **kwargs)
                    for backend in backends)


def iter_paid_engines(basify=True, **kwargs):
    """Yield tuple containing names for the two paid backends, openai and
    gooseai. Mostly useful for testing purposes.
    """
    yield from iter_engine_names('openai', 'gooseai', basify=basify, **kwargs)


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

    def __init__(self, tasks=(), verbose=True,
                 prompt_dir=C.root/'data/prompts', skip_tasks=()):
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
        # Maps task name to query kwargs. Prompt templates have not yet been
        # evaluated at this point (i.e. still contain literal '{}' where values
        # will later be filled in).
        self.prompt_dir = Path(prompt_dir)
        self.prompts = self._load_templates(tasks, skip_tasks)
        self.log_path = GPT.logger.path

    def _load_templates(self, tasks, skip_tasks=()):
        """Load template and default hyperparameters for each prompt.

        Parameters
        ----------
        tasks: Iterable[str]
            If empty, we load all available prompts in the data/prompts
            directory that are not in skip_tasks.
        skip_tasks: Iterable[str]
            If provided, skip these tasks. This overrides `tasks`.

        Returns
        -------
        dict[str, dict]: Maps task name to dict of hyperparameters
        (including the prompt template).
        """
        name2kwargs = {}
        paths = (self.prompt_dir/f'{t}.yaml' for t in tasks) if tasks \
            else self.prompt_dir.iterdir()
        if skip_tasks: paths = (p for p in paths if p.stem not in skip_tasks)
        for path in paths:
            if path.suffix != '.yaml':
                if tasks:
                    warnings.warn(f'Skipping {path} due to unrecognized name.')
                continue
                if path.stem == 'conversation':
                    warnings.warn(
                        'Skipping loading "conversation" prompt. Use '
                        'ConversationManager if you want to use this.'
                    )
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
            example, maybe our 'tldr' task usually uses engine=3, but we want
            to test it with engine=0 once. We can pass in that value here,
            but future method calls without that kwarg will use the expected
            default behavior of engine=3.

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
        prompt = self.format_prompt(task, kwargs.pop('prompt'), text=text)
        if debug:
            print('prompt:\n' + prompt)
            print(spacer())
            print('kwargs:\n', kwargs)
            print(spacer())
            print('fully resolved kwargs:\n',
                  dict(bound_args(query_gpt3, [], kwargs)))
            return
        return GPT.query(prompt, **kwargs)

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
        if 'prompt' in kwargs:
            raise RuntimeError('Arg "prompt" should not be in query kwargs. '
                               'It will be constructed within this method and '
                               'passing it in will override the new version.')
        if 'mock_func' in kwargs:
            raise ValueError(
                'Encountered unexpected argument "mock_func". This was a part '
                'of the Jabberwocky 1.0 API but is no longer used.'
            )

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

        if fully_resolved:
            kwargs = dict(bound_args(GPT.query, [], kwargs))
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
        res = self.format_prompt(task, template, text)
        if print_:
            print(res)
        else:
            return res

    def format_prompt(self, task, template, text=''):
        # Handle tasks that require some special handling to integrate
        # user-provided text into the prompt.
        if text:
            formatter = TASK2FORMATTER.get(task, default_formatter)
            res = formatter(template, text)
        else:
            res = template
        return res

    def __repr__(self):
        return f'{type(self).__name__}({", ".join(map(repr, self.prompts))})'

    def __iter__(self):
        """Iterates over prompt names (strings)."""
        return iter(self.prompts)


class ConversationManager:
    """Similar to PromptManager but designed for ongoing conversations. This
    currently references just a single prompt: conversation.
    """

    img_exts = {'.jpg', '.jpeg', '.png'}

    def __init__(self, names=True, custom_names=True, data_dir=C.root/'data',
                 backup_image='data/misc/unknown_person.png',
                 turn_window=3, me='me', prompt='conversation_transcript',
                 load_qa_pipe=False, qa_model_name=None, verbose=True):
        """
        Parameters
        ----------
        names: Iterable[str] or bool
            Optionally specify 1 or more personas to load. These should be
            pretty-formatted, e.g. "Barack Obama" rather than "barack_obama".
            Do not include periods (e.g. "TJ Dillashaw" rather than
            "T.J. Dillashaw"). You can also pass in True to load all available
            personas or False to avoid loading any personas.
        custom_names: Iterable[str] or bool
            Same as `names` but for customer personas (those that are
            user-defined rather than auto-generated).
        data_dir: str or Path
            Data dir where all necessary subdirs will be created and accessed.
        backup_image: str or Path
            Path to default image to use when we can't find one for the
            current persona.
        turn_window: int
            Number of most recent speaker "turns" to include in each query.
            4 means we'll include 2 user turns and 2 gpt3 responses (at most -
            early in the conversation we'll necessarily use fewer until we
            hit those numbers). Note that user_turns will be >= gpt3_turns
            regardless of your choice of window because the last turn must be
            from the user in order to prompt gpt3 for a response. We enforce
            a limit because gpt3 can't handle infinitely long sequences so
            we must do something to allow long conversations. Some have
            reported success with summarizing past portions of the conversation
            but I wanted to start with something relatively simple.
        me: str
            What to call the user in the conversation. This will be title-cased
            for you automatically.
        """
        assert 1 <= turn_window <= 20, 'turn_window should be in [1, 20].'
        self.verbose = verbose

        if custom_names and me == 'me':
            warnings.warn(
                'You\'ve chosen to load >=1 custom personas but you are using '
                'me="me". Some custom personas expect you to set `me` to your '
                'name. Stop phrases may not work as intended if you do not '
                'override conv.me.'
            )

        # We'll be adding in the user's newest turn separately from accessing
        # their historical turns so we need to subtract 1 from both of these.
        self.user_turn_window = int(np.ceil(turn_window / 2)) - 1
        self.gpt3_turn_window = turn_window - self.user_turn_window - 1

        # Set directories for data storage, logging, etc.
        self.backup_image = Path(backup_image)
        self.data_dir = Path(data_dir)
        # We often access this like `self.person_dir[is_custom]` where
        # is_custom is a bool specifying whether the persona is
        # custom-generated, i.e. not a real person with a wikipedia page.
        # This works because indexing with False works like 0 and indexing with
        # True works like 1.
        self.persona_dir = [self.data_dir/'conversation_personas',
                            self.data_dir/'conversation_personas_custom']
        self.conversation_dir = self.data_dir/'conversations'
        self.log_path = GPT.logger.path
        for dir_ in (*self.persona_dir, self.conversation_dir):
            os.makedirs(dir_, exist_ok=True)

        # Load prompt, default query kwargs, and existing personas. Set self.me
        # after loading _kwargs since the setter must update them.
        self._prompt = prompt
        self._kwargs = load_prompt(prompt, verbose=self.verbose)
        self._base_prompt = self._kwargs.pop('prompt')
        self.me = me

        # This QA pipeline can be used to extract nationality from wiki
        # summaries for auto-generated personas. The import + model loading
        # can take several seconds so I've set it to False, at least during
        # development (jupyter autoreload makes slow imports very annoying).
        # Do this before we add the Albert Einstein persona below.
        self.qa_pipe = self._load_qa_pipe(load_qa_pipe, qa_model_name)

        # Populated by _load_personas().
        self.name2base = {}
        self.name2meta = {}
        self.name2kwargs = {}

        # Custom personas are loaded last so they override default personas.
        self._load_personas(names, is_custom=False)
        self._load_personas(custom_names, is_custom=True)
        # We provide at least 1 default persona. This also ensures that
        # self.current is populated upfront with the right keys (values are
        # empty strings, but we want it to know what the fields should be,
        # e.g. "summary", "gender", etc.).
        if not self.name2meta:
            self.add_persona('Albert Einstein')

        # Keep self.current definition after loading persona(s) so we know
        # what meta fields there are.
        # self.current stores attributes about the persona currently being
        # spoken to. It's populated when we load a persona and cleared
        # when we end a conversation. The 'persona' key must be manually added
        # the first time since it's not in our meta dicts. It contains the
        # processed name (i.e. lowercase w/ underscores).
        self.current = {k: '' for k in next(iter(self.name2meta.values()))}
        self.current['persona'] = ''
        self.user_turns = []
        self.gpt3_turns = []
        self.cached_query = ''

    def is_active(self):
        """Check if there's an active conversation taking place.

        Returns
        -------
        bool: True if a conversation is underway, False otherwise.
        """
        return bool(self.current['persona'])

    def _load_qa_pipe(self, do_load, model_name=None):
        """Load QA pipeline to help extract nationality from wikipedia summary
        for auto-generated personas. Alternative is to use html parsing - this
        avoids the upfront cost of loading the model but seems to be slower
        per person (I think the wikipedia lib doesn't fetch the html by default
        but the html parsing method obviously requires it).

        Parameters
        ----------
        do_load: bool
            If True, load QA model. Otherwise self.qa_pipe will be None and
            html parsing will be used to infer nationality.
        model_name: str or None
            Transformers QA model to use. If None, the dfeault is used
            (distilbert-base-cased-distilled-squad as of 5/22/22).

        Returns
        -------
        transformers.QuestionAnsweringPipeline or None
        """
        # We place some imports here because loading them at the module level
        # slows things down a lot when using jupyter autoreload.
        if not do_load:
            return

        from transformers import pipeline

        if model_name:
            from transformers import AutoModelForQuestionAnswering, \
                AutoTokenizer

            model = AutoModelForQuestionAnswering.from_pretrained(model_name)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
        else:
            model = tokenizer = None

        return pipeline('question-answering', model=model, tokenizer=tokenizer)

    def _load_personas(self, names, is_custom=False):
        """Load any stored summaries and image paths of existing personas."""
        if names is False:
            return
        if names is True:
            names = [path.stem for path in
                     self.persona_dir[is_custom].iterdir()
                     if path.is_dir()]
        for name in names:
            try:
                self.update_persona_dicts(self.process_name(name),
                                          is_custom=is_custom)
            except:
                warnings.warn(f'Could not load files for {name}.')

    def start_conversation(self, name, download_if_necessary=False,
                           resume_text='', resume_path=''):
        """Prepare for a conversation with a specified persona. We need to
        track several variables during the conversation.

        Parameters
        ----------
        name: str
            The persona you wish to converse with. This should be a
            pretty-formatted name (no underscores).
        download_if_necessary: bool
            If True and the requested persona doesn't exist, this will download
            the necessary materials and add that persona to their "rolodex".
        resume_text: str
            We can resume a conversation by passing in a transcript of an old
            conversation (e.g. saved with ConversationManager.save()).
        resume_path: str
            We can resume a conversation by specifying a path to a saved
            conversation transcript generated by ConversationManager.save().
            This is an alternative to resume_text - you should specify at most
            one of these two args.

        Returns
        -------
        tuple: Processed name (str; with underscores), running prompt
        (str; basically just the persona's summary at this point), and path to
        the persona's image (Path).
        """
        if name not in self:
            if not download_if_necessary:
                raise KeyError(f'{name} persona not available. You can set '
                               'download_if_necessary=True if you wish to '
                               'construct a new persona.')
            _ = self.add_persona(name, return_data=True)
        self.end_conversation()

        if resume_path:
            assert not resume_text, \
                ('You provided both resume_text and resume_path. You should '
                 'provide at most one of these args.')
            resume_text = load(resume_path)
        if resume_text:
            self.user_turns, self.gpt3_turns = self._load_conv_from_str(
                resume_text, name=name
            )
        processed_name = self.process_name(name)
        self.current = {'persona': processed_name,
                        **self.name2meta[processed_name]}
        return self.current

    def _load_conv_from_str(self, full_text, name):
        """Extract user turns and gpt turns from a previously transcribed
        conversation. This allows start_conversation() to resume an old
        conversation.

        Parameters
        ----------
        full_text: str
            Result of ConversationManager.full_summary() or text loaded from a
            file generated by ConversationManager.save().
        name: str
            Pretty-formatted name, e.g. "Sam Altman".

        Returns
        -------
        tuple[list[str]]: First list contains user turns (strings). Second list
        contains gpt turns (strings).
        """
        # We override `name` in loop but we need to use original name later.
        requested_name = name
        name2turns = defaultdict(list)
        prev_name = None
        for row in full_text.splitlines()[2:]:
            if not row: continue
            if ': ' not in ' '.join(row.split(' ')[:4]):
                turns = name2turns[prev_name]
                if not turns:
                    raise RuntimeError(
                        'Failed to parse conversation string. Trying to '
                        'attach line to previous speaker turn but there is '
                        'no previous turn.'
                    )
                turns[-1] = turns[-1] + '\n\n' + row
                continue
            name, _, turn = row.partition(': ')
            name2turns[name].append(turn)
            prev_name = name
        if len(name2turns) != 2:
            raise RuntimeError(
                'Expected conversation to have two participants but found: '
                f'{list(name2turns)}.'
            )
        me, partner = name2turns
        if me != self.me:
            warnings.warn(
                f'"Me" is named "{me}" in loaded conversation but '
                f'self.me is "{self.me}". We\'ll be sticking with "{self.me}".'
            )
        if partner != requested_name:
            warnings.warn(
                f'Conversation partner is named "{partner}" in loaded '
                f'conversation but requested name is "{requested_name}". '
                f'We\'ll assume "{requested_name}" is correct.'
            )
        return list(name2turns.values())

    def end_conversation(self, fname=None):
        """Resets several variables when a conversation is over. This is also
        called automatically at the beginning of start_conversation in case you
        forgot to close the previous conversation.

        Parameters
        ----------
        fname: str
            If non-empty, the transcript of this conversation will be saved to
            a text file by this name. This should not be a full path - it will
            automatically be saved in the manager's conversation_dir.
        """
        if fname: self.save_conversation(fname)
        self.cached_query = ''
        self.user_turns.clear()
        self.gpt3_turns.clear()
        self.current = {k: '' for k, v in self.current.items()}

    def save_conversation(self, fname):
        if not self.user_turns:
            raise RuntimeError('No conversation to save.')
        save(self.full_conversation(), self.conversation_dir/fname)

    def add_persona(self, name, summary=None, img_path=None, gender=None,
                    nationality=None, is_custom=False, return_data=False,
                    wiki_tags=()):
        """Download materials for a new persona. This saves their wikipedia
        summary and profile photo in a directory with their name inside the
        persona_dir.

        Parameters
        ----------
        name: str
            Pretty-formatted name (e.g. 'Barack Obama', not 'barack_obama') of
             the persona you'd like to add.
        summary: str or None
            Only specify when is_custom is True: in that case, this should be
            a 1-3 sentence description of the persona you're creating. This
            can be used to define both their personality and the tone of the
            conversation.
        img_path: str or Path or None
            Only specify when is_custom is True: in that case, you can
            optionally pass in the local file path for an image of the persona
            you're creating.
        gender: str or None
            Only specify when is_custom is True: in that case, pass in a str in
            ('F', 'M'). Otherwise leave as None.
        is_custom: bool
            True if you want to manually define a custom persona (usually
            someone who doesn't exist or isn't well known, otherwise we'd
            construct them in the normal way from wikipedia data). When True,
            you must pass in a summary and gender (img_path is optional). When
            False, you must no pass in any of those three parameters.
        return_data: bool
            When True, return dictionary containing summary str and other
            metadata. Otherwise return None.
        """
        if (summary or img_path or gender or nationality) and not is_custom:
            raise ValueError('You can only pass in summary/img_path/gender'
                             '/nationality for a custom persona.')

        processed_name = self.process_name(name)
        dir_ = self.persona_dir[is_custom]/processed_name
        tmp_dir = '/tmp'
        # Case where we seem to already have the data for this persona.
        if dir_.is_dir():
            if summary or img_path or gender or nationality:
                raise ValueError(
                    'Do not pass in summary/img_path/gender/nationality for '
                    'a persona that already exists.'
                )
            meta = self.update_persona_dicts(
                processed_name, return_values=True, is_custom=is_custom
            )
            meta = meta._asdict()
        else:
            if is_custom:
                if not (summary and gender):
                    raise ValueError(
                        'Must provide a summary and gender for a custom '
                        'persona that does not yet exist locally.'
                    )
                meta = {'summary': summary,
                        'gender': gender,
                        'img_path': img_path,
                        'nationality': nationality}
            else:
                # Autogenerate persona metadata.
                meta = wiki_data(name, img_dir=tmp_dir, fname='profile',
                                 tags=wiki_tags, qa_pipe=self.qa_pipe)
                meta = meta._asdict()
                # Update processed_name and dir_ in case we made a typo, e.g.
                # "Apollo Ohno" instead of "Apolo Ohno". Note that at this
                # stage, the img_path includes the tmp_dir, but this will be
                # updated before saving the metadata file.
                processed_name = self.process_name(meta.pop('name'))
                dir_ = self.persona_dir[is_custom]/processed_name
                os.makedirs(dir_, exist_ok=True)

            # We always need to move an image (in custom mode, either the
            # backup image or a user-specified img_path from another dir -
            # remember this is the case where no persona dir exists yet). In
            # non-custom mode, we either move the downloaded image from the
            # /tmp dir or we copy the backup image to the new persona dir.
            # Be careful with logic: Path('abc') != 'abc', and if we convert
            # img_path to a Path immediately, we'd interpret Path('') as
            # truthy.
            src_path = meta['img_path'] or self.backup_image
            img_path = dir_/f'profile{Path(src_path).suffix}'
            try:
                # Don't delete user's custom image in its original path or the
                # backup image, but do move any downloaded image.
                if str(src_path) != str(img_path):
                    if is_custom or str(src_path) == str(self.backup_image):
                        shutil.copy2(src_path, img_path)
                    else:
                        shutil.move(src_path, img_path)
            except FileNotFoundError as e:
                # Clean up newly-added dir otherwise this will affect
                # subsequent attempts to run this method.
                shutil.rmtree(dir_)
                raise e

            # Do this after moving the image file (if necessary) so meta.yaml
            # contains the correct image path.
            meta.update(img_path=str(img_path))
            save_yaml(meta, dir_/'meta.yaml')

            # It's an empty string if we fail to download an image in
            # non-custom mode, or None if we choose not to pass in a path in
            # custom mode.
            self.update_persona_dicts(processed_name, is_custom=is_custom)
        if return_data: return meta

    def update_persona_dicts(self, processed_name, return_values=False,
                             is_custom=False):
        """Helper to update our various name2{something} dicts.

        Returns
        -------
        htools.Results: Contains summary str and misc metadata such as gender,
        img_path, etc.
        """
        dir_ = self.persona_dir[is_custom]/processed_name
        meta = load_yaml(dir_/'meta.yaml')
        summary = meta.pop('summary')
        self.name2meta[processed_name] = meta
        self.name2base[processed_name] = self._base_prompt.format(
            name=self.process_name(processed_name, inverse=True),
            summary=summary
        )
        if processed_name not in self.name2kwargs:
            self.name2kwargs[processed_name] = {}
        if return_values:
            return Results(summary=summary, **meta)

    def set_default_kwargs(self, name='', force=False, **kwargs):
        try:
            name = self.process_name(name) or self.current['persona']
            # This will be an empty str if there is no current speaker.
            assert name
        except:
            raise ValueError(
                'You did not specify a name and there is no active speaker '
                'so it\'s not clear who you\'re trying to set default '
                'kwargs for. Either start a conversation or pass in a name '
                'explicitly.'
            )
        unrecognized = set(kwargs) - set(self._kwargs)
        if unrecognized:
            msg = f'The following kwargs are not usually specified in our ' \
                  f'conversational settings: {unrecognized}. '
            if force:
                msg += 'Adding them anyway because force=True.'
            else:
                kwargs = {k: v for k, v in kwargs.items() if
                          k not in unrecognized}
                msg += 'Pass in force=True if you really want to add these. ' \
                       f'Right now we will only set {list(kwargs)}.'
            warnings.warn(msg)
        self.name2kwargs[name].update(kwargs)

    def clear_default_kwargs(self, *names, all_=False):
        if all_:
            if names:
                warnings.warn('Clearing default kwargs for all names because '
                              'all_=True.')
            names = self.personas(pretty=False)
        elif names:
            names = [self.process_name(name) for name in names]
        else:
            if not self.is_active():
                raise RuntimeError(
                    'You passed in zero names and specified all_=False and '
                    'there is no active conversation, so we don\'t know which '
                    'settings you want to clear. Try passing in one or more '
                    'names, setting all_=True, or starting a conversation '
                    'first.'
                )
            names = [self.current['persona']]
        for name in names:
            self.name2kwargs[name].clear()

    def persona_exists_locally(self, name):
        """Check if a persona's info files are already available locally.

        Parameters
        ----------
        name: str
            Pretty-formatted name, e.g. "Albert Einstein" rather than
            "albert_einstein".

        Returns
        -------
        bool
        """
        processed_name = self.process_name(name)
        for dir_ in self.persona_dir:
            dir_ = dir_/processed_name
            if dir_.is_dir() and 'meta.yaml' in [path.name
                                                 for path in dir_.iterdir()]:
                return True
        return False

    def process_name(self, name, inverse=False):
        """Convert a name to snake case (by default) or from snake case to
        pretty format (title case, no underscores) if inverse=True.
        The default method also removes periods (technically, you
        shouldn't be including these to begin with, but it's not a serious
        enough violation to throw an error) but note that inverse will NOT
        re-insert periods.

        Parameters
        ----------
        name: str
        inverse: bool
            If False, convert a pretty-formatted string to snake case (replace
            spaces with underscores and remove periods). If True, perform
            the inverse operation, converting a snake case string to pretty
            format. Periods are never added back in.
        """
        if inverse:
            # To pretty format.
            return name.replace('_', ' ').title().replace('Dr ', 'Dr. ')
        return name.lower().replace(' ', '_').replace('.', '')

    def nearest_persona(self, name):
        """Find the nearest persona in the list of available personas.

        Parmeters
        ---------
        name: str
            Can be pretty formatted (e.g. John Smith) or not (e.g. john_smith).
            This also determines whether the output name will be
            pretty-formatted or not.

        Returns
        -------
        tuple[str, float]: First value is name in self.personas(). If the input
        name has an underscore, this will be the snake_cased version; if not,
        it will be the pretty-formatted version. The second value is a float
        in [0.0, 1.0] measuring similarity to the input, where higher is
        better. Empirically, a threshold around 0.8 may be reasonable for
        identifying a match.
        """
        processed = '_' in name
        if processed:
            name = name.lower()
        else:
            name = name.title()
        res = process.extractOne(
            name,
            [person for person in self.personas(pretty=not processed)]
        )
        return res[0], res[1] / 100.0

    def personas(self, pretty=True, sort=True):
        """Quick way to see a list of all available personas.

        Returns
        -------
        list[str]
        """
        names = list(self.name2base)
        if pretty: names = [self.process_name(name, True) for name in names]
        if sort: names = sorted(names)
        return names

    def kwargs(self, name='', fully_resolved=True, return_prompt=False,
               extra_kwargs=None, **kwargs):
        # Name param should be pretty version, i.e. no underscores. Only
        # needed when return_prompt is True.
        if 'prompt' in kwargs:
            raise RuntimeError(
                'Arg "prompt" should not be in query kwargs. It will be '
                'constructed within this method and passing it in will '
                'override the new version.'
            )
        if 'mock_func' in kwargs:
            raise ValueError(
                'Encountered unexpected argument mock_func. This was a part '
                'of the Jabberwocky 1.0 API but is no longer used.'
            )
        if return_prompt and not name:
            warnings.warn('You set return_prompt=True but we cannot return a '
                          'prompt because you didn\'t provide a `name`.')

        # If no name is specified AND no conversation is active, this resolves
        # to an empty string.
        name = self.process_name(name or self.current['persona'])
        kwargs = {**self._kwargs, **self.name2kwargs.get(name, {}), **kwargs}
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

        if fully_resolved:
            kwargs = dict(bound_args(GPT.query, [], kwargs))

        # Note: this just returns the base prompt. In query(), we use the
        # format_prompt() method to attach the conversational turns.
        if name and return_prompt:
            kwargs['prompt'] = self.name2base[name]
        return kwargs

    def query_later(self, text):
        """Cache a user turn to query later. This should NOT start with
        'Me:' or include the persona summary.

        Parameters
        ----------
        text: str
        """
        self.cached_query = text.strip()

    def query(self, text=None, debug=False, extra_kwargs=None, **kwargs):
        """
        Parameters
        ----------
        text: None or str
            The thing you'd like to say next. This should not start with
            "Me: " - that will be inserted automatically. This should also not
            include the wikipedia summary.

        Returns
        -------
        tuple[str]: Tuple of (prompt, response), just like all gpt3 queries.
        if not self.current['persona']:
            raise RuntimeError('You must call the `start_conversation` '
                               'method before making a query.')
        """
        # In the same spirit as our handling of kwargs here, passing in a text
        # arg will override a cached query if one exists.
        text = text or self.cached_query
        if not text:
            raise ValueError('You must pass in an argument for text when no '
                             'query has been previously cached.')

        kwargs = self.kwargs(fully_resolved=False, return_prompt=False,
                             extra_kwargs=extra_kwargs, **kwargs)
        if kwargs.get('stream', False): kwargs['strip_output'] = False
        prompt = self.format_prompt(user_text=text)
        if debug:
            print('prompt:\n' + prompt)
            print(spacer())
            print('kwargs:\n', kwargs)
            print(spacer())
            print('fully resolved kwargs:\n',
                  dict(bound_args(query_gpt3, [], kwargs)))
            return

        # Update turns after query in case something goes wrong and it
        # doesn't actually execute.
        res = GPT.query(prompt, **kwargs)
        self.user_turns.append(text.strip())
        self.cached_query = ''

        if not kwargs.get('stream', False):
            # In v2 api, response is a tuple of (text, full) where text is a
            # list of strings. At least right now, ConversationManager only
            # supports a single prompt with single completion.
            self.gpt3_turns.append(res[0][0])
            return res
        return hooked_generator(res, self.turn_hook)

    def turn_hook(self, item, i, is_post=False):
        text = item[0]
        if is_post:
            self.gpt3_turns[-1] = self.gpt3_turns[-1].strip()
        elif i == 0:
            self.gpt3_turns.append(text)
        else:
            self.gpt3_turns[-1] += text

    def _format_prompt(self, user_text='', do_full=False,
                       include_trailing_name=True, include_summary=True):
        """Convert a new user turn to a fully-specified prompt. This also
        provides the core logic that allows us to reconstruct a full
        conversation from turns without also storing and updating a separate
        full_conv attribute (as we briefly did while developing support for
        longer conversations).

        Parameters
        ----------
        user_text
        do_full: bool
            If True, return the full conversation. If False, return a new
            prompt (if the conversation is longer than a few turns, this will
            not include the beginning of the conversation, though it will still
            include the wikipedia summary).
        include_trailing_name:
            If True, include current persona's name (followed by a colon) on a
            new line at the end of the prompt. This should be True when you
            plan to use the output as a prompt.

        Returns
        -------
        str
        """
        current_persona = self.current['persona']
        if not current_persona:
            raise RuntimeError('No persona loaded. Have you started a '
                               'conversation?')
        if not do_full and not user_text:
            raise RuntimeError('user_text must be provided when '
                               'do_full=False.')

        pretty_name = self.process_name(current_persona, inverse=True)
        if do_full:
            user_turns = list(self.user_turns)
            if user_text: user_turns.append(user_text)
            gpt3_turns = self.gpt3_turns
        else:
            user_turns = (self.user_turns[-self.user_turn_window:]
                          + [user_text.strip()])
            gpt3_turns = self.gpt3_turns[-self.gpt3_turn_window:]

        if len(user_turns) - len(gpt3_turns) not in (0, 1):
            raise RuntimeError(
                f'Mismatched turn counts: user has {len(user_turns)} and gpt3'
                f' has {len(gpt3_turns)} turns.'
            )
        user_turns = [f'{self.me}: {turn}' for turn in user_turns]
        # Strip gpt3 turns to be safe since streaming mode only strips them
        # once the full query completes, and GUI uses full_conversation
        # property while query is still in progress.
        gpt3_turns = [f'{pretty_name}: {turn.strip()}' for turn in gpt3_turns]
        ordered = [user_turns, gpt3_turns]
        if len(gpt3_turns) == len(user_turns) and not do_full:
            ordered = reversed(ordered)
        interleaved = filter(None, flatten(zip_longest(*ordered)))
        prompt = '\n\n'.join(interleaved)
        if include_summary:
            prompt = f'{self.name2base[current_persona]}\n\n' + prompt
        if not include_trailing_name:
            return prompt
        return f'{prompt}\n\n{self.process_name(current_persona, True)}:'

    def format_prompt(self, user_text, include_trailing_name=True,
                      include_summary=True):
        """Single formatted prompt including wiki bio and last few speaker
        turns (number depends on turn window).

        Parameters
        ----------
        user_text: str
            Does not start with "Me:".
        include_trailing_name: bool

        Returns
        -------
        str
        """
        return self._format_prompt(
            user_text, do_full=False,
            include_trailing_name=include_trailing_name,
            include_summary=include_summary
        )

    def full_conversation(self, include_summary=True):
        """Reconstruct full conversation from turns. Note that at the moment,
        this will still return a value (the wiki bio) when no turns have
        occurred.

        Returns
        -------
        str
        """
        return self._format_prompt(do_full=True, include_trailing_name=False,
                                   include_summary=include_summary)

    @property
    def me(self):
        return self._me

    @me.setter
    def me(self, me):
        # Make sure to get old value before setting new one.
        try:
            old = self.me
        except AttributeError:
            # Dummy value since _kwargs doesn't really need to be updated in
            # this case.
            old = me
        self._me = me.title()
        self._kwargs['stop'] = [text if text != f'\n\n{old}:'
                                else f'\n\n{self.me}:'
                                for text in self._kwargs['stop']]

    @me.deleter
    def me(self):
        raise RuntimeError('ConversationPersona attribute `me` cannot be '
                           'deleted.')

    @contextmanager
    def converse(self, name, fname='', download_if_necessary=False):
        """Wanted to provide context manager even though we can't easily use it
        in GUI.

        Parameters
        ----------
        name: str
            Pretty-formatted name of persona to talk to.
        fname: str
            If not empty, this will be where manager will save the full
            transcript when done. This should just be the file name, not the
            full path.
        download_if_necessary: bool
        """
        try:
            _ = self.start_conversation(name, download_if_necessary)
            yield
        finally:
            self.end_conversation(fname=fname)

    @staticmethod
    def format_conversation(text, gpt_color='black', me='me'):
        """Add some string formatting to a conversation: display names and
        initial summary in bold and optionally change the color of
        gpt3-generated responses. This doesn't print anything - it just
        returns an updated string.

        Warning: because this is a static method, the `me` attribute must be
        manually specified if you want to set it to something other than "me".

        Parameters
        ----------
        text: str or tuple[str]
            Conversation consisting of an initial summary followed by
            exchanges between "Me" and `name`. You can pass in a single string
            or the (prompt, response) tuple returned by self.query().
        gpt_color: str
            Control the color of your gpt3 conversational partner's lines.
            Default is the same as the user, but you can change it if you want
            more distinct outputs.

        Returns
        -------
        str
        """
        def _format(line, color='black'):
            if not line: return line
            name, _, line = line.partition(':')
            # Bold's stop character also resets color so we need to color the
            # chunks separately.
            return colored(bold(name + ':'), color) + colored(line, color)

        if listlike(text): text = ' '.join(text)
        summary, *lines = text.splitlines()
        name = [name for name, n in
                Counter(line.split(':')[0]
                        for line in lines if ':' in line).most_common(2)
                if name != me.title()][0]
        formatted_lines = [bold(summary)]
        prev_is_me = True
        for line in lines:
            if line.startswith(name + ':'):
                line = _format(line, gpt_color)
                prev_is_me = False
            elif line.startswith(f'{me.title()}: ') or prev_is_me:
                line = _format(line)
                prev_is_me = True
            formatted_lines.append(line)
        return '\n'.join(formatted_lines)

    def __contains__(self, name):
        """
        Parameters
        ----------
        name: str
            Pretty-formatted version (no underscores).
        """
        return self.process_name(name) in self.name2base

    def __len__(self):
        return len(self.name2base)


def print_response(prompt, response, sep=' '):
    """Print gpt3 prompt and response. Prompt is in bold to make it easier to
    distinguish between them.

    Parameters
    ----------
    prompt: str
    response: str
        The text returned by gpt3.
    sep: str
        Placed between prompt and response. Defaults to a space because we
        often apply str.strip to prompt and response automatically.
    """
    print(bold(prompt), end=sep)
    print(response)


def load_prompt(name, prompt='', rstrip=True, verbose=True,
                prompt_dir=C.root/'data/prompts', base_url=''):
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

    Note: 'reminder' field in prompt config file is optional.

    Parameters
    ----------
    name: str
        Name of subdirectory in data/prompts. Ex: 'simplify_ml'
    prompt: str or dict
        Additional input to be inserted into the prompt template. For example,
        our tldr template prompt is "{}\n\ntl;dr:". We need to pass in text
        to summarize (this replaces the brackets like in a python f-string).
        A str input is required if the prompt template expects 1 variable, like
        the tldr example above. Prompts with multiple variables expect a dict
        (e.g. we might have the template 'I am {name} and I am from {state}.',
        then pass in prompt={'name': Sam, 'state': 'California'}).
    rstrip: bool
        This is a safety measure to prevent us from accidentally leaving a
        trailing space after the end of the prompt (which leads to worse gpt3
        completions). We let the user turn it off in case a prompt requires it.
    verbose: bool
        If True, this will print a message on loading if one is specified in
        the prompt config file. This can be some kind of reminder or usage
        note.
    base_url: str
        If provided, this should be a url that allows us to load a remote
        prompt by making an http request to base_url.format(name).
        The code to load remote prompts expects that this will resolve to a
        URL for viewing a raw file hosted on github. You can find this type of
        URL by going to a git repo containing your prompts, clicking on a
        prompt yaml file, clicking the button to view the raw version, then
        copying the URL and replacing the file name (but keeping the file
        extension) with '{}'. See jabberwocky.config.C.prompt_base_url
        for an example:
        (
            'https://raw.githubusercontent.com/hdmamin/'
            'jabberwocky/main/data/prompts/{}.yaml'
        )
        If provided, this will override the prompt_dir arg.

    Returns
    -------
    dict: Keys are all kwargs for query_gpt3(). You may want to override some
    of these at times, but they at least provide reasonable defaults. Some are
    more important than others: for example, a 'stop' value will likely always
    be relevant, while 'max_tokens' or 'engine' may depend on the specific
    usage.
    """
    if base_url:
        r = requests.get(base_url.format(name))
        kwargs = yaml.safe_load(r.content.decode())
        prompt_fmt = kwargs.pop('prompt')
    else:
        path = Path(prompt_dir)/name
        # V1 style prompt files are stored in dir, v2 style are stored in
        # single config.
        if path.is_dir():
            prompt_fmt = load(path / 'prompt.txt')
            kwargs = load_yaml(path / 'config.yaml')
        else:
            kwargs = load_yaml(f'{path}.yaml')
            prompt_fmt = kwargs.pop('prompt')

    # If no prompt is passed in, we load the template and leave variable
    # imputation for later.
    if prompt:
        formatter = TASK2FORMATTER.get(name, default_formatter)
        prompt = formatter(prompt_fmt, prompt)
    else:
        prompt = prompt_fmt

    # Vim adds trailing newline, which can hurt gpt3 quality.
    if rstrip: prompt = prompt.rstrip()

    # Pyyaml seems to escape newlines (probably other special characters too
    # but this is the only one I've used here, I think. Newline chars can be
    # useful in stop terms because I often use them to distinguish between
    # different examples in a prompt.
    # 4/20/22 update: not seeing any '\\n' even if I remove the str replacement
    # below. A bit hesitant to remove this now since it's not causing any harm
    # and I must have observed this behavior at one point to feel the need to
    # add and document it.
    if 'stop' in kwargs:
        kwargs['stop'] = [x.replace('\\n', '\n') for x in kwargs['stop']]
    kwargs['prompt'] = prompt
    # 5/30/22 update: adding a "version" field that currently is a single int
    # ONLY tracking changes to the text template, not the hyperparameters.
    # Currently leaving this in the kwargs returned by load_prompt. GPT._query
    # pops it and logs it in the "meta" dict for each query. Considering
    # eventually putting doc/reminder/version/etc. under 1 "meta" field in the
    # yaml file, but holding off for now since I don't have a super clear idea
    # of how I want the versioning to work or how to use it. All query funcs
    # accept kwargs anyway so even if it did slip past the GPT._query pop
    # (which it shouldn't), there shouldn't really be a problem.
    kwargs.pop('doc', None)
    msg = kwargs.pop('reminder', None)
    if msg and verbose: print(f'{name}: {msg}{spacer()}')
    return kwargs


def punctuate_mock_func(prompt, random_punct=True, sentence_len=15,
                        *args, **kwargs):
    """Mimic punctuate task by splitting a piece of unpunctuated text into
    chunks of constant length and inserting periods and capitalization in the
    corresponding places. Pass to query_gpt_3 as mock_func.

    Parameters
    ----------
    prompt
    random_punct
    sentence_len

    args and kwargs are just included for compatiblity with mock_fn interface.

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
    return prompt, text


###############################################################################
#  Formatter functions accept text_fmt (a str prompt template where variables
#  have not yet been inserted), prompt (a str or dict containing the variables
# to insert into text_fmt), and optional kwargs. It should return a fully
# resolved prompt str ready to be sent to a gpt-like model. default_formatter
# should be general enough to handle most cases, but you could define new
# format functions for different prompts - just make sure to add them to
# the TASK2FORMATTER dict below.
###############################################################################

def default_formatter(text_fmt, prompt, **kwargs):
    """Default formatter to insert args into a prompt template. Called by
    load_prompt.

    Parameters
    ----------
    text_fmt: str
        Prompt template. If it only contains 1 variable, it should be unnamed
        (sample template: 'My name is {}.'). If there are multiple, each
        must be named (sample template:
        'My name is {name} and I am {age} years old'). Positional args are not
        allowed.
    prompt: str or dict
        Str if it's a 1-variable prompt, dict if it uses multiple variables.
        If it contains zero variables, either are fine.
    kwargs: any
        Just included for compatibility with other formatters. These should not
        be passed in when using this formatter.

    Returns
    -------
    str
    """
    if kwargs:
        raise ValueError('Kwargs should not be provided when using the '
                         f'default prompt formatter. You passed in {kwargs}.')

    if isinstance(prompt, str):
        return text_fmt.format(prompt)
    if isinstance(prompt, Mapping):
        return text_fmt.format(**prompt)
    raise ValueError(f'Prompt should either be a str or a dict. '
                     f'You passed in {type(prompt)}.')


@deprecated
def conversation_formatter(text_fmt, prompt, **kwargs):
    """Integrate user-provided values into the pre-existing template. This is
    necessary (rather than a simple str.format call, as the other tasks so far
    use) because we have multiple input fields (name, message, summary) and we
    must do some extra work to prepare them (extract name from message,
    call wikipedia api to obtain summary).

    Parameters
    ----------
    text_fmt: str
        The template loaded from prompt.txt file. This has the fields "name",
        "summary", and "message" that must be filled in.
    prompt: str
        Should have keys "name" (person to converse with) and "message" (your
        first message to them).
    kwargs: any
        Additional kwargs to pass to wiki_data function.
        Ex: min_similarity, tags.

    Returns
    -------
    str: Finalized prompt where user-provided values have been integrated into
    the pre-existing template.
    """
    assert prompt.startswith('Hi '), 'Prompt must start with "Hi ".'
    name = sent_tokenize(prompt)[0].replace('Hi ', '').rstrip('.')
    # Still trying to think of a good backward-compatible way to pass img_path
    # on to caller. Thinking it may be simplest to either change logic to
    # always download to some constant temp filename, or to just make GUI load
    # the most recently created/changed file in the temp dir.
    summary, *_ = wiki_data(name, **kwargs)
    return text_fmt.format(name=name, summary=summary, message=prompt)


def query_kwargs_grid(verbose=True):
    """Generator that yields kwargs for gpt.query() to generate all variations
    of responses (n_prompts = 1 or > 1, n_completions = 1 or > 1,
    stream = True or False). There are therefore 8 different sets of kwargs.
    Other kwargs (max_tokens, engine) are set such that they match the
    saved responses generated by s01.
    """
    txts = ['Yesterday was', 'How many']
    # Just like keys (multi_in, multi_out, stream) in pickled MOCKS dict in
    # this same module. Start with easier cases (multiple prompts/completions
    # and/or streaming complicate things a bit).
    for multi_in in (False, True):
        for multi_out in (False, True):
            for stream in (False, True):
                prompt = txts if multi_in else txts[0]
                nc = 1 + multi_out
                if verbose:
                    print(f'np>1: {multi_in}\nnc>1: {multi_out}'
                          f'\nstream: {stream}')
                yield dict(prompt=prompt,
                           n=nc,
                           stream=stream,
                           engine=0,
                           max_tokens=3,
                           logprobs=3)


def convert_old_prompt_files(prompt_dir, dest_dir=None):
    """Convert a directory prompts using the jabberwocky v1 format/conventions
    (where a prompt was defined by a dir (named with the name of the prompt,
    e.g. data/prompts/summarize/) containing two files: a config.yaml
    containing model hyperparameters and a prompt.txt containing the
    natural language prompt) to the new prompt format
    (consisting of a single prompt.yaml file named something like
    data/prompts/summarize.yaml). I think I initially had trouble getting
    special characters and multiline strings to play nicely together in yaml
    files, but I think they're working now so I want to convert them. This
    doesn't delete the old dir because I want to encourage keeping the old
    prompts around for maintainability.

    Parameters
    ----------
    prompt_dir: str or Path
    dest_dir: str or Path or None
        If None, dest_dir will be set to the same dir as prompt_dir.
    """
    dest_dir = Path(dest_dir or prompt_dir)
    os.makedirs(dest_dir, exist_ok=True)
    for dir_ in Path(prompt_dir).iterdir():
        if not dir_.is_dir(): continue
        print(f'Generating yaml for {dir_}...')
        with open(dir_/'prompt.txt', 'r') as f:
            text = f.read()
        with open(dir_/'config.yaml', 'r') as f:
            cfg = f.read()
        cfg += f'prompt: |-\n{text.strip()}'.replace('\n', '\n    ')
        with open(dest_dir/f'{dir_.name}.yaml', 'w') as f:
            f.write(cfg)


@valuecheck
def upload_openai_files(purpose:('search', 'answers', 'classifications'), *,
                        path2meta=None, doc2meta=None, out_path=None):
    """Upload files to openai for semantic search. I believe it's (potentially)
    slightly cheaper to pre-upload this way than to pass in all documents
    at search time.

    Parameters
    ----------
    path2meta: None or dict[str or Path, dict] or list[str or Path]
        Maps file paths (each containing text) to a dict of metadata. If using
        purpose='classifications', that metadata dict must include a key
        'label'. Always provide either this OR doc2meta (that is the same but
        instead of file paths, you provide strings which are the "documents"
        themselves. If listlike is provided, there will be no metadata.
    doc2meta: None or dict[str, dict] or list[str or Path]
    purpose: str
        Think this determines how openai indexes input files
        (Basically, ask yourself: do you plan to use these for semantic search,
        question answering, etc.?).
    out_path: str or Path or None
        If provided, this should be a path to output the resulting file to
        (should be jsonlines format). The resulting file will essentially
        contain a dictionary on each line with keys "text" (pointing to a str
        containing the content of one of the files) and "metadata" containing
        the local file path.

    Returns
    -------
    OpenAIObject: Dict-like object with various metadata about the upload.

    Ex:
    <OpenAIObject file id=file-ymFoe5t2hW5v9VJrJUmPacCE at 0x125a2ed00> JSON: {
        "bytes": 8011,
        "created_at": 1650598850,
        "filename": "/tmp/5cCjHuNcTmMBAOX4HIgvFCUPCXBMCf.jsonlines",
        "id": "file-ymFoe5t2hW5v9VJrJUmPacCE",
        "object": "file",
        "purpose": "search",
        "status": "uploaded",
        "status_details": null
    }

    Note that you can call openai.File.delete(res.id) on the result of this
    function to delete the file you uploaded.
    """
    assert bool(path2meta) + bool(doc2meta) == 1, \
        'Pass in either path2meta OR doc2meta.'
    if path2meta:
        def get_data(path, meta):
            with open(path, 'r') as f:
                text = f.read()
            return {'text': text, 'metadata': {**meta, 'path': str(obj)}}
        items = path2meta
    else:
        def get_data(doc, meta):
            return {'text': doc, 'metadata': meta}
        items = doc2meta

    if not isinstance(items, Mapping):
        items = {item: {} for item in items}

    rm_after = not out_path
    out_path = out_path or f'/tmp/{random_str(30)}.jsonlines'
    touch(out_path)
    with open(out_path, mode='r+') as outfile:
        for obj, meta in items.items():
            data = get_data(obj, meta)
            if purpose == 'classifications':
                data['label'] = data['metadata'].pop('label')
            json.dump(data, outfile)
            outfile.write('\n')

        outfile.seek(0)
        res = openai.File.create(file=outfile, purpose=purpose)
    if rm_after: os.remove(out_path)
    return res


# In theory, this can be used to map task name to prompt formatter function.
# This used to map 'conversation' to conversation_formatter, but I now
# recommend against using that interface - it was only useful with
# PromptManager, but ConversationManager now provides a better version of that
# for the conversation use case.
TASK2FORMATTER = {}

# I figure if we're importing these functions, we'll need to authenticate.
openai_auth()