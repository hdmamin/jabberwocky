"""Utility functions for interacting with the gpt3 api."""

import banana_dev as banana
from collections.abc import Iterable, Mapping
from collections import Counter
from contextlib import contextmanager
from itertools import zip_longest
import json
from nltk.tokenize import sent_tokenize
import numpy as np
import openai
import os
from pathlib import Path
import requests
import shutil
import sys
import warnings

from htools import load, select, bound_args, spacer, valuecheck, tolist, save,\
    listlike, Results, flatten, add_docstring, func_name, params
from jabberwocky.config import C
from jabberwocky.external_data import wiki_data
from jabberwocky.utils import strip, bold, load_yaml, colored, \
    hooked_generator, load_api_key, with_signature, squeeze, stream_response


HF_API_KEY = load_api_key('huggingface')
BANANA_API_KEY = load_api_key('banana')
MOCK_RESPONSE = [load(C.mock_stream_paths[False]),
                 load(C.mock_stream_paths[True])]


def truncate_at_first_stop(text, stop_phrases, finish_reason='',
                           trunc_full=True, trunc_partial=True,
                           partial_pct=.8, partial_n=4):
    """Remove stop phrases from gpt completions, since some backends either
    don't provide this functionality or do but leave the trailing stop word
    attached. We also provide the option to try to detect if the completion
    finished mid-stopword and remove it if so. E.g. with a stopword of
    "\n\nMe:", a completion that ends with "\n\nMe" and finished due to
    length constraints likely should be stripped.

    Parameters
    ----------
    text: str
        GPT completion.
    stop_phrases: list[str]
        One or more words/phrases that signify that a completion should end.
    finish_reason: str
        Reason a completion ended (if "length", this means it was cut short due
        to a max_token limit and therefore is at risk for a possible partial
        stop_phrase remaining at the end). Currently only provided by
        gpt3/gooseai backends.
    trunc_full: bool
        If True, assume the backend does not automatically remove the
        truncating stop word and instead stops AFTERWARDS (or not at all).
        As of 4/2/22, this should only be False for the openai backend.
    trunc_partial: bool
        If True and finish_reason is "length", we'll try to truncate if the
        completion ENDS WITH a partial stop word.
    partial_pct: float
        When truncating partial stop phrases, this is the percent
        of a stop phrase that much be matched. Should lie in (0, 1).
    partial_n: int
        When truncating partial stop phrases, this the minimum number of
        characters that must match. (We might want to ensure that super short
        matches don't qualify.)

    Returns
    -------
    str: Input text truncated right before the first stop phrase (if any), and
    possibly before a partial stop phrase depending on user-specified options.
    """
    if trunc_full:
        idx = [idx for idx in map(text.find, stop_phrases) if idx >= 0]
        stop_idx = min(idx or [None])
        text = text[:stop_idx]

    # If the completion was cut short due to length AND the completion ends
    # with the majority of a stop phrase, we infer that this should be stripped
    # from the end. This rule won't be perfect but it seems like a decent bet.
    if trunc_partial and finish_reason == 'length':
        for phrase in stop_phrases:
            thresh = max(int(round(partial_pct * len(phrase))), partial_n)
            chunk = phrase[:thresh]
            if text.endswith(chunk):
                warnings.warn(
                    'Guessing that truncation is reasonable because '
                    'finish_reason="length" and completion ends with a '
                    'partial stop phrase.'
                )
                return text.rpartition(chunk)[0]
    return text


def query_gpt_j(prompt, temperature=0.7, max_tokens=50, **kwargs):
    """Queries free GPT-J API. GPT-J has 6 billion parameters and is, roughly
    speaking, the open-source equivalent of Curie (3/19/22 update: size sounds
    more like Babbage actually). It was trained on more
    code than GPT3 though so it may do surprisingly well at those kinds of
    tasks. This function should be usable as a mock_func argument in
    query_gpt_3.

    API uptime may be questionable though. There's an accompanying front end
    here:
    http://api.vicgalle.net:8000/

    Parameters
    ----------
    prompt: str
    temperature: float
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
              'top_p': kwargs.pop('top_p', 1.0)}

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
    return res['text'], res


# def query_gpt_j(prompt, temperature=0.7, max_tokens=50, **kwargs):
#     """Queries free GPT-J API. GPT-J has 6 billion parameters and is, roughly
#     speaking, the open-source equivalent of Curie (3/19/22 update: size sounds
#     more like Babbage actually). It was trained on more
#     code than GPT3 though so it may do surprisingly well at those kinds of
#     tasks. This function should be usable as a mock_func argument in
#     query_gpt_3.
#
#     API uptime may be questionable though. There's an accompanying front end
#     here:
#     http://api.vicgalle.net:8000/
#
#     Parameters
#     ----------
#     prompt: str
#     temperature: float
#     max_tokens: int
#     kwargs: any
#         Only supported options are top_p (float) and stop (Iterable[str]).
#         Notice that stream mode is not supported.
#
#     Returns
#     -------
#     tuple[str]: Prompt, response.
#     """
#     params = {'context': prompt,
#               'token_max_length': max_tokens,
#               'temperature': temperature,
#               'top_p': kwargs.pop('top_p', 1.0)}
#
#     # Ensure that we end up with a list AND that stop is still Falsy if user
#     # explicitly passes in stop=None.
#     stop = tolist(kwargs.pop('stop', None) or [])
#     if stop: params['stop_sequence'] = stop[0]
#
#     # Must keep this after the block of stop-related logic above.
#     if kwargs:
#         warnings.warn(f'GPT-J api does not support other kwargs: {kwargs}')
#
#     try:
#         res = requests.post('http://api.vicgalle.net:5000/generate',
#                             params=params)
#         res.raise_for_status()
#     except Exception as e:
#         raise MockFunctionException(str(e)) from None
#     res = res.json()
#
#     # Endpoint doesn't support multiple stop sequences so we have to
#     # postprocess. Even with a single stop sequence, it includes it while gpt3
#     # and my gpt-neo function exclude it, so we need to handle that here.
#     idx = min([i for i in map(res['text'].find, stop) if i >= 0] or [None])
#     completion = res['text'][:idx]
#     return res['prompt'], completion


@valuecheck
def query_gpt_huggingface(
        prompt, engine_i=0, temperature=1.0, repetition_penalty=None,
        max_tokens=50, top_k=None, top_p=None, n=1, **kwargs
):
    """Query EleuetherAI gpt models using the Huggingface API. This was called
    query_gpt_neo in a former version of the library (which is used by the
    GUI) but the API now hosts a GPT-J model as well so I renamed it.

    Parameters
    ----------
    prompt: str
    engine_i: int
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
    # TODO update
    # Currently returns List[str], List[dict] where index i in each list gives
    # us the ith completion. We always pass in a single prompt.

    tuple or iterator: When stream=False, we return a tuple where the first
    item is the prompt (str) and the second is the response text(str). If
    return_full is True, a third item consisting of the whole response object
    is returned as well. When stream=True, we return an iterator where each
    step contains a single token. This will either be the text response alone
    (str) or a tuple of (text, response) if return_full is True. Unlike in
    non-streaming mode, we don't return the prompt - that seems less
    appropriate for many time steps.

    Returns
    -------
    tuple[str]: Prompt, response tuple, just like query_gpt_3().
    """
    if not isinstance(prompt, str):
        raise TypeError(f'Prompt must be str, not {type(prompt)}.')

    # Hardcode backend in case we use this function outside of the
    # GPTBackend.query wrapper.
    engine = GPTBackend.engine(engine_i, backend='huggingface')

    # Docs say we can return up to 256 tokens but API sometimes throws errors
    # if we go above 250.
    headers = {'Authorization':
                   f'Bearer api_{HF_API_KEY}'}
    # Notice the names don't always align with parameter names - I wanted
    # those to be more consistent with query_gpt3() function. Also notice
    # that types matter: if Huggingface expects a float but gets an int, we'll
    # get an error.
    if repetition_penalty is not None:
        repetition_penalty = float(repetition_penalty)
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


# @valuecheck
# def query_gpt_huggingface(
#         prompt, engine_i=0, temperature=1.0, repetition_penalty=None,
#         max_tokens=50, top_k=None, top_p=None, **kwargs
# ):
#     """Query EleuetherAI gpt models using the Huggingface API. This was called
#     query_gpt_neo in a former version of the library (which is used by the
#     GUI) but the API now hosts a GPT-J model as well so I renamed it.
#
#     Parameters
#     ----------
#     prompt: str
#     engine_i: int
#         Determines which Huggingface model API to query. See
#         config.C.backend_engines['huggingface'].
#         Those names refer to the number of
#         parameters in the model, where bigger models generally produce higher
#         quality results but may be slower (in addition to the actual inference
#         being slower to produce, the better models are also more popular so the
#         API is hit with more requests).
#     temperature: float
#         Between 0 and 1. 0-0.4 is good for straightforward informational
#         queries (e.g. reformatting, writing business emails) while 0.7-1 is
#         good for more creative works.
#     top_k: None or int
#         Kind of like top_p in that smaller values may produce more
#         sensible but less creative responses. While top_p limits options to
#         a cumulative percentage, top_k limits it to a discrete number of
#         top choices.
#     top_p: None or float
#         Value in [0.0, 1.0] if provided. Kind of like temperature in that
#         smaller values may produce more sensible but less creative responses.
#     repetition_penalty
#     max_tokens: int
#         Sets max response length. One token is ~.75 words. IIRC 250 is the max
#         they support.
#     kwargs: any
#         Just lets us absorb extra kwargs when used in place of query_gpt3().
#
#     Returns
#     -------
#     tuple or iterator: When stream=False, we return a tuple where the first
#     item is the prompt (str) and the second is the response text(str). If
#     return_full is True, a third item consisting of the whole response object
#     is returned as well. When stream=True, we return an iterator where each
#     step contains a single token. This will either be the text response alone
#     (str) or a tuple of (text, response) if return_full is True. Unlike in
#     non-streaming mode, we don't return the prompt - that seems less
#     appropriate for many time steps.
#     Returns
#     -------
#     tuple[str]: Prompt, response tuple, just like query_gpt_3().
#     """
#     engine = GPTBackend.engine(engine_i)
#
#     # Docs say we can return up to 256 tokens but API sometimes throws errors
#     # if we go above 250.
#     headers = {'Authorization':
#                f'Bearer api_{HF_API_KEY}'}
#     # Notice the names don't always align with parameter names - I wanted
#     # those to be more consistent with query_gpt3() function. Also notice
#     # that types matter: if Huggingface expects a float but gets an int, we'll
#     # get an error.
#     if repetition_penalty is not None:
#         repetition_penalty = float(repetition_penalty)
#     stop = tolist(kwargs.pop('stop', []))
#     if kwargs:
#         warnings.warn('query_gpt_huggingface received unused kwargs '
#                       f'{kwargs}.')
#
#     data = {'inputs': prompt,
#             'parameters': {'top_k': top_k, 'top_p': top_p,
#                            'temperature': float(temperature),
#                            'max_new_tokens': min(max_tokens, 250),
#                            'repetition_penalty': repetition_penalty,
#                            'return_full_text': False}}
#     url = f'https://api-inference.huggingface.co/models/EleutherAI/{engine}'
#     try:
#         # Put the request itself inside try too in case of timeout.
#         r = requests.post(url, headers=headers, data=json.dumps(data))
#         r.raise_for_status()
#     except requests.HTTPError as e:
#         raise MockFunctionException(str(e)) from None
#
#     # Huggingface doesn't natively provide the `stop` parameter that OpenAI
#     # does so we have to do this manually.
#     res = r.json()[0]['generated_text']
#     if stop:
#         idx = [idx for idx in map(res.find, stop) if idx >= 0]
#         stop_idx = min(idx) if idx else None
#         res = res[:stop_idx]
#     return prompt, res


def query_gpt_repeat(prompt, n=1, **kwargs):
    """Mock func that just returns the prompt as the response."""
    if kwargs:
        warnings.warn(f'Unused kwargs {kwargs} received by query_gpt_repeat.')
    return [prompt for _ in range(n)], [{} for _ in range(n)]


# def query_gpt3(prompt, engine_i=0, temperature=0.7, top_p=1.0,
#                frequency_penalty=0.0, presence_penalty=0.0,
#                max_tokens=50, logprobs=None, n=1, stream=False,
#                logit_bias=None, mock=False, return_full=False,
#                strip_output=True,
#                mock_func=None, mock_mode:('raise', 'warn', 'ignore')='raise',
#                **kwargs):
#     """Convenience function to query gpt3. Mostly serves 2 purposes:
#     1. Build in some mocking functionality for cheaper/free testing.
#     2. Explicitly add some parameters and descriptions to the function
#     docstring, since openai.Completion.create does not include most kwargs.
#
#     Parameters
#     ----------
#     prompt: str
#     engine_i: int
#         Corresponds to engines defined in config, where 0 is the cheapest, 3
#         is the most expensive, etc.
#     temperature: float
#         Between 0 and 1. 0-0.4 is good for straightforward informational
#         queries (e.g. reformatting, writing business emails) while 0.7-1 is
#         good for more creative works.
#     top_p: float
#         Value in (0.0, 1.0] that limits the model to sample from tokens making
#         up the top_p percent combined. I.e. higher values allow for more
#         creativity (like high temperature) and low values are closer to argmax
#         sampling (like low temperature). API recommends setting a sub-maximal
#         value for at most one of this and temperature, not both.
#     frequency_penalty: float
#         Value in [-2.0, 2.0] where larger (more positive) values more heavily
#         penalize words that have already occurred frequently in the text.
#         Usually reasonable to keep this in [0, 1].
#     presence_penalty: float
#         Value in [-2.0, 2.0] where larger (more positive) values more heavily
#         penalize words that have already occurred in the text. Usually
#         reasonable to keep this in [0, 1].
#     max_tokens: int
#         Sets max response length. One token is ~.75 words.
#     logprobs: int or None
#         Get log probabilities for top n candidates at each time step. This
#         will only be useful if you set return_full=True.
#     n: int
#         Number of possible completions to return. Careful: values > 1 can add
#         up quickly w.r.t. cost.
#     stream: bool
#         If True, return an iterator instead of a str/tuple. See the returns
#         section as the output is slightly different. I believe each chunk
#         returns one token when stream is True.
#     logit_bias: dict or None
#         If provided, should map string(s) (NUMERIC INDEX of word tokens,
#         not the tokens themselves) to ints between -100 and 100 (inclusive?).
#         Values in (-1, 1) should be used to nudge the model, while larger
#         values can effectively ban/compel the model to use certain words.
#     mock: bool
#         If True and no mock_func is provided, return a saved sample response
#         instead of hitting the API
#         in order to save tokens. Note that your other gpt3 kwargs
#         (max_tokens, logprobs, kwargs) will be ignored. return_full will be
#         respected since it affects the number of items returned - it's not a
#         kwarg passed to the actual query function. Text is surrounded by
#         <MOCK></MOCK> tags to make it obvious when mock is True (it's easy to
#         forget to change the value of mock when switching back and forth).
#     return_full: bool
#         If True, return a third item which is the full response object.
#         Otherwise we just return the prompt and response text.
#     strip_output: bool
#         If True, strip text returned by gpt3. Without this, many prompts have a
#         leading space and/or trailing newlines due to the way examples are
#         formatted.
#     mock_func: None or function
#         You can provide a function here that accepts the prompt
#         and returns something which will be used as the mock text.
#         As of version 1.1.0, this automatically sets mock=True. Sample use
#         case: when punctuating a transcript, the text realignment process may
#         raise an error when loading a saved mock response. Therefore, we may
#         want to write a mock_func that extracts the new input portion of the
#         prompt (discarding instructions and examples). This option is
#         unavailable in stream mode. [3/24/22: that last line might not be true
#         with gooseAI. Need to investigate further.]
#     mock_mode: str
#         Determines what to do if using mock mode and mock_func is not None and
#         it fails. Either 'raise' an error, 'warn' the user and proceed with the
#         saved response, or silently proceed with the saved response.
#     kwargs: any
#         Additional kwargs to pass to gpt3 or mock_func. Most useful openai
#         API kwargs are already in the docstring so mostly intended for the
#         latter.
#
#     Returns
#     -------
#     tuple or iterator: When stream=False, we return a tuple where the first
#     item is the prompt (str) and the second is the response text(str). If
#     return_full is True, a third item consisting of the whole response object
#     is returned as well. When stream=True, we return an iterator where each
#     step contains a single token. This will either be the text response alone
#     (str) or a tuple of (text, response) if return_full is True. Unlike in
#     non-streaming mode, we don't return the prompt - that seems less
#     appropriate for many time steps.
#     """
#     if stream and strip_output:
#         warnings.warn('strip_output is automatically set to False when stream '
#                       'is True. It would be impossible to correctly '
#                       'reconstruct outputs otherwise.')
#     if temperature < 1 and top_p < 1:
#         warnings.warn('You set both temperature and top_p to values < 1. '
#                       'API recommends setting only one of these to '
#                       'sub-maximal value.')
#     if logprobs and not return_full:
#         warnings.warn('You set logprobs to a nonzero value but '
#                       'return_full=False. If you want to access the logprobs, '
#                       'you should set return_full=True.')
#
#     # Realized GPT-J was often being called without setting mock=True, leading
#     # to unexpected charges 😬. Trying to prevent this in the future.
#     mock = mock or bool(mock_func)
#     kwargs = {
#         **kwargs,
#         'prompt': prompt,
#         'temperature': temperature,
#         'top_p': top_p,
#         'frequency_penalty': frequency_penalty,
#         'presence_penalty': presence_penalty,
#         'max_tokens': max_tokens,
#         'logprobs': logprobs,
#         'n': n,
#         'stream': stream,
#         'logit_bias': logit_bias
#     }
#     if mock:
#         res = MOCK_RESPONSE[stream]
#         if mock_func:
#             if stream:
#                 raise NotImplementedError('mock_func unavailable when '
#                                           'stream=True.')
#
#             # Replace text with results of mocked call if possible. Mock_funcs
#             # return response as the second item, just like this function, so
#             # we can also use them in place of it instead of specifying a
#             # mock_func parameter.
#             try:
#                 res.choices[0].text = mock_func(engine_i=engine_i, **kwargs)[1]
#             except MockFunctionException as e:
#                 if mock_mode == 'raise':
#                     raise e
#                 elif mock_mode == 'warn':
#                     warnings.warn(str(e))
#     else:
#         res = openai.Completion.create(engine=GPTBackend.engine(engine_i),
#                                        **kwargs)
#
#     if stream:
#         # In this case, we get 1 response for each new token. Zip does maintain
#         # lazy evaluation.
#         texts = (chunk.choices[0].text for chunk in res)
#         return zip(texts, res) if return_full else texts
#     else:
#         output = (prompt, strip(res.choices[0].text, strip_output), res)
#         return output if return_full else output[:-1]


def query_gpt3(prompt, engine_i=0, temperature=0.7, top_p=1.0,
               frequency_penalty=0.0, presence_penalty=0.0, max_tokens=50,
               logprobs=None, n=1, stream=False, logit_bias=None, **kwargs):
    """Convenience function to query gpt3. Mostly serves 2 purposes:
    1. Build in some mocking functionality for cheaper/free testing.
    2. Explicitly add some parameters and descriptions to the function
    docstring, since openai.Completion.create does not include most kwargs.

    # TODO: update docs

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
    top_p: float
        Value in (0.0, 1.0] that limits the model to sample from tokens making
        up the top_p percent combined. I.e. higher values allow for more
        creativity (like high temperature) and low values are closer to argmax
        sampling (like low temperature). API recommends setting a sub-maximal
        value for at most one of this and temperature, not both.
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
        Additional kwargs to pass to gpt3. Should rarely be necessary.

    Returns
    -------
    # TODO: update
    tuple[list[str], List[dict]]

    tuple or iterator: When stream=False, we return a tuple where the first
    item is the prompt (str) and the second is the response text(str). If
    return_full is True, a third item consisting of the whole response object
    is returned as well. When stream=True, we return an iterator where each
    step contains a single token. This will either be the text response alone
    (str) or a tuple of (text, response) if return_full is True. Unlike in
    non-streaming mode, we don't return the prompt - that seems less
    appropriate for many time steps.
    """
    if stream and n > 1:
        raise RuntimeError('Stream=True and n>1 not supported.')

    #     if stream and strip_output:
    #         warnings.warn('strip_output is automatically set to False when stream '
    #                       'is True. It would be impossible to correctly '
    #                       'reconstruct outputs otherwise.')
    if temperature < 1 and top_p < 1:
        warnings.warn('You set both temperature and top_p to values < 1. '
                      'API recommends setting only one of these to '
                      'sub-maximal value.')
    #     if logprobs and not return_full:
    #         warnings.warn('You set logprobs to a nonzero value but '
    #                       'return_full=False. If you want to access the logprobs, '
    #                       'you should set return_full=True.')

    res = openai.Completion.create(
        engine=GPTBackend.engine(engine_i),
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

    # Extract text and return. Zip maintains lazy evaluation.
    if stream:
        # Each item in zipped object is (str, dict-like).
        texts = (chunk.choices[0].text for chunk in res)
        chunks = (dict(chunk.choices[0]) for chunk in res)
        return zip(texts, chunks)

    # Structure: (List[str], List[dict-like])
    return [row.text for row in res.choices], \
           [dict(choice) for choice in res.choices]


@add_docstring(query_gpt3)
def query_gpt_banana(prompt, temperature=.8, max_tokens=50, top_p=.8,
                     top_k=False, **kwargs):
    """Free gptj access. Unclear which version of the model they provide -
    guessing 6B params? The query_gpt3 docstring is included below for
    convenience, but that doesn't mean all of its parameters are supported;
    only those visible in the signature are (all kwargs are ignored here). For
    example, this does not natively support stop phrases or n_prompts > 1 or
    n_completions > 1.
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
    try:
        res = banana.run(api_key=BANANA_API_KEY, model_key='gptj',
                         model_inputs=params)
        # Do this in separate line to make it easier for humans to parse error
        # messages.
        res = res['modelOutputs'][0]
        return res['input'], res['output']
    except Exception as e:
        raise MockFunctionException(str(e)) from None


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

    # Only include backends here that actually should change the
    # openai.api_base value (these will probably be backends that require no
    # or minimal mock_funcs).
    name2base = {
        'openai': 'https://api.openai.com',
        'gooseai': 'https://api.goose.ai/v1',
    }

    # Order matters: keep openai first so name2key initialization works.
    name2func = {
        'openai': query_gpt3,
        'gooseai': query_gpt3,
        'huggingface': query_gpt_huggingface,
        'hobby': query_gpt_j,
        'repeat': query_gpt_repeat,
        'banana': query_gpt_banana
    }

    # Names of backends that perform stop word truncation how we want (i.e.
    # allow us to specify stop phrases AND truncate before the phrase rather
    # than after, if we encounter one).
    skip_trunc = {'openai'}

    name2key = {}
    for name in name2func:
        if name in {'hobby', 'repeat'}:
            name2key[name] = f'<{name.upper()} BACKEND: FAKE API KEY>'
        else:
            name2key[name] = load_api_key(name)

    def __init__(self):
        self.new_name = ''
        self.old_name = ''
        self.old_key = ''

    def __call__(self, name):
        """__enter__ can't take arguments so we need to specify this here.
        Notice that name is auto-lowercased and spaces are removed.
        """
        new_name = name.lower().replace(' ', '')
        if new_name not in self.name2func:
            raise ValueError(f'Invalid name {name}. Valid options are: '
                             f'{list(self.name2func)}')

        self.new_name = new_name
        self.old_name = self.current()
        return self

    def __enter__(self):
        """Change backend to the one specified in __call__, which is
        automatically called first when using `with` syntax.
        """
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
        print(f'Switching  backend back to "{self.old_name}".')
        openai.api_key = self.old_key
        if self.old_name in self.name2base:
            openai.api_base = self.name2base[self.old_name]
        openai.curr_name = self.old_name
        self.clear()

    @classmethod
    def ls(cls):
        """Print current state of the backend: api_base, api_key, and 
        mock_func. Mostly useful for debugging and sanity checks.
        """
        print('\nBase:', openai.api_base)
        print('Key:', openai.api_key)
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
        # huggingface mock_func technically uses a global).
        return openai.api_key

    @classmethod
    def engine(cls, engine_i, backend=None):
        """Get appropriate engine name depending on current api backend and
        selected engine_i.

        Parameters
        ----------
        engine_i: int
            Number from 0-3 (inclusive) specifying which model to use. The two
            backends *should* perform similar for values of 0-2, but openai's
            3 (davinci, 175 billion parameters) is a much bigger model than
            gooseai's 3 (NEO-X, 20 billion parameters). Mostly used in
            query_gpt3().
        backend: str or None
            If provided, should be the name of a backend (e.g. 'huggingface'
            or any of the keys in GPTBackend.backends()).

        Returns
        -------
        str: Name of an engine, e.g. "davinci" if we're in openai mode or
        "gpt-neo-20b" if we're in gooseai mode.
        """
        engines = C.backend_engines[backend or cls.current()]

        # Adds better error message if user passes in a number too big for the
        # current backend.
        try:
            return engines[engine_i]
        except IndexError:
            raise ValueError(f'Encountered invalid engine_i value: {engine_i}.'
                             f'Should be one of {list(range(len(engines)))} '
                             f'when using backend {cls.current()}.')

    # Decorator order matters - doesn't work if we flip these.
    @classmethod
    @with_signature(query_gpt3)
    @add_docstring(query_gpt3)
    def query(cls, prompt, strip_output=True, log_path=None, **kwargs):
        """

        Parameters
        ----------
        prompt
        strip_output
        log_path
        kwargs

        Returns
        -------
        list[str, dict]
        """
        if not isinstance(prompt, str):
            raise NotImplementedError(
                f'Prompt must be str, not {type(prompt).__name__}.'
            )

        # Keep trunc_full definition here so we can provide warnings if user
        # is in stream mode.
        query_func = cls._get_query_func()
        trunc_full = cls.current() not in cls.skip_trunc
        stream = kwargs.get('stream', False)
        if stream:
            if kwargs.get('n', 1) > 1:
                raise ValueError('Stream mode not supported when n > 1.')
            if strip_output:
                warnings.warn('strip_output=True is not supported in stream '
                              'mode. Automatically setting it to False.')
                strip_output = True
            if trunc_full:
                warnings.warn(
                    'Streaming mode does not support manual truncation of '
                    'stop phrases and your current backend has limited '
                    'support for truncation.'
                )

        # V2 library no longer supports user passing in mock_func. We want to
        # remove this from the kwargs we pass to our actual function.
        kwargs_func = kwargs.pop('mock_func', None)
        if kwargs_func:
            raise ValueError(
                f'Encountered unexpected mock_func {kwargs_func} with this '
                'interface. This was part of the v1 library but is no longer '
                'supported.'
            )

        kwargs['prompt'] = prompt
        cls._log_query_kwargs(log_path, query_func=query_func, **kwargs)

        # Possibly easier for caller to check for errors this way? Mostly a
        # holdover from v1 library design, but I'm not 100% sure if the
        # benefits still hold given the new design.
        try:
            text, full_response = query_func(**kwargs)
        except Exception as e:
            raise MockFunctionException(str(e)) from None
        if stream:
            if 'stream' in params(query_func):
                return text, full_response
            # Squeeze is necessary to support both valid query response types:
            # (str, dict) and (list[str], list[dict]).
            return stream_response(*squeeze(text, full_response, n=1))

        # Manually check for stop phrases because most backends either don't
        # or truncate AFTER the stop phrase which is rarely what we want.
        stop = kwargs.get('stop', [])
        clean_text = []
        # tolist doesn't know how to handle dicts so we check explicitly.
        if not listlike(text):
            text = tolist(text)
            full_response = [full_response]
        for text_, resp_ in zip(text, full_response):
            text_ = truncate_at_first_stop(
                text_,
                stop_phrases=stop,
                finish_reason=resp_.get('finish_reason', ''),
                trunc_full=trunc_full,
                trunc_partial=True
            )
            clean_text.append(strip(text_, strip_output))

        return squeeze(clean_text, full_response, n=kwargs.get('n', 1))

    @classmethod
    def _log_query_kwargs(cls, path, query_func=None, **kwargs):
        """Log kwargs for troubleshooting purposes."""
        if path:
            # Meta key is used to store any info we want to log but that should
            # not be passed to the actual query_gpt3 call.
            kwargs['meta'] = {
                'backend_name': cls.current(),
                'query_func': func_name(query_func) if query_func else None
            }
            save(kwargs, path, verbose=False)

    def __repr__(self):
        return f'{func_name(self)} <current_name: {self.current()}>'


class MockFunctionException(Exception):
    """Allow all mock query functions to return a common exception."""


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

    def __init__(self, tasks=(), verbose=True, log_dir='data/logs',
                 skip_tasks=()):
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
        self.prompts = self._load_templates(tasks, skip_tasks)
        self.log_dir = log_dir
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            self.log_path = Path(log_dir)/'query_kwargs.json'
        else:
            self.log_path = None

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
        dir_ = Path('data/prompts')
        paths = (dir_/t for t in tasks) if tasks else dir_.iterdir()
        if skip_tasks: paths = (p for p in paths if p.stem not in skip_tasks)
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
        prompt = self.format_prompt(task, kwargs.pop('prompt'), text=text)
        if debug:
            print('prompt:\n' + prompt)
            print(spacer())
            print('kwargs:\n', kwargs)
            print(spacer())
            print('fully resolved kwargs:\n',
                  dict(bound_args(query_gpt3, [], kwargs)))
            return
        return GPTBackend.query(prompt, log_path=self.log_path, **kwargs)

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
        mock_func = GPTBackend._get_query_func()
        kwargs_mock = kwargs.get('mock_func', None)
        # If user doesn't pass in a mock_func explicitly, this shouldn't raise
        # an error even if our backend will automatically use one.
        if kwargs_mock and kwargs_mock != mock_func:
            raise ValueError(
                f'Encountered unexpected mock_func {kwargs_mock}. The current '
                f'backend expects a mock_func of {mock_func}. Note: you '
                'typically shouldn\'t pass in mock_func explicitly since '
                'GPTBackend handles this for you. '
                '(Technically, we do allow this due to the way PromptManager '
                'and ConversationManager implement kwargs() methods).'
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

        if fully_resolved: kwargs = dict(bound_args(query_gpt3, [], kwargs))
        kwargs['mock_func'] = mock_func
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
        # Handle tasks like "conversation" where we need to do some special
        # handling to integrate user-provided text into the prompt.
        if text:
            formatter = TASK2FORMATTER.get(task)
            if formatter:
                res = formatter(template, text)
            else:
                res = template.format(text)
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

    def __init__(self, names=(), custom_names=(), data_dir='./data',
                 backup_image='data/misc/unknown_person.png',
                 turn_window=3, me='me', verbose=True):

        """
        Parameters
        ----------
        names: Iterble[str]
            Optionally specify 1 or more personas to load. These should be
            pretty-formatted, e.g. "Barack Obama" rather than "barack_obama".
            If None are provided, all available personas will be loaded.
            Do not include periods (e.g. "TJ Dillashaw" rather than
            "T.J. Dillashaw").
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
            a limit because gpt3 can't handle indefinitely long sequences so
            we must do something to allow long conversations. Some have
            reported success with summarizing past portions of the conversation
            but I wanted to start with something relatively simple.
        me: str
            What to call the user in the conversation. This will be title-cased
            for you automatically.
        """
        assert 1 <= turn_window <= 20, 'turn_window should be in [1, 20].'
        self.verbose = verbose

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
        self.log_dir = self.data_dir/'logs'
        self.log_path = Path(self.log_dir)/'conversation_query_kwargs.json'
        for dir_ in (*self.persona_dir, self.conversation_dir, self.log_dir):
            os.makedirs(dir_, exist_ok=True)

        # These attributes will be updated when we load a persona and cleared
        # when we end a conversation. current_persona is the processed name
        # (i.e. lowercase w/ underscores).
        self.current_persona = ''
        self.current_summary = ''
        self.current_img_path = ''
        self.current_gender = ''
        self.cached_query = ''
        self.user_turns = []
        self.gpt3_turns = []

        # Load prompt, default query kwargs, and existing personas. Set self.me
        # after loading _kwargs since the setter must update them.
        self._kwargs = load_prompt('conversation', verbose=self.verbose)
        self._base_prompt = self._kwargs.pop('prompt')
        self.me = me

        # Populated by _load_personas().
        self.name2img_path = {}
        self.name2base = {}
        self.name2gender = {}

        # Custom personas are loaded last so they override default personas.
        self._load_personas(names, is_custom=False)
        self._load_personas(custom_names, is_custom=True)

    def _load_personas(self, names, is_custom=False):
        """Load any stored summaries and image paths of existing personas."""
        names = names or [path.stem for path in
                          self.persona_dir[is_custom].iterdir()
                          if path.is_dir()]
        for name in names:
            try:
                self.update_persona_dicts(self.process_name(name),
                                          is_custom=is_custom)
            except:
                warnings.warn(f'Could not load files for {name}.')

    def start_conversation(self, name, download_if_necessary=False):
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

        processed_name = self.process_name(name)
        self.current_persona = processed_name
        self.current_img_path = self.name2img_path[processed_name]
        self.current_gender = self.name2gender[processed_name]
        # This one is not returned. Info would be a bit repetitive.
        self.current_summary = self._name2summary(processed_name)
        return (self.current_persona,
                self.current_img_path,
                self.current_gender)

    def _name2summary(self, name):
        if '_' not in name: name = self.process_name(name)
        base = self.name2base[name]
        intro = sent_tokenize(base)[0]
        return base.replace(intro, '').strip()

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
        self.current_summary = ''
        self.current_persona = ''
        self.current_img_path = ''
        self.current_gender = ''
        self.cached_query = ''
        self.user_turns.clear()
        self.gpt3_turns.clear()

    def save_conversation(self, fname):
        if not self.user_turns:
            raise RuntimeError('No conversation to save.')
        save(self.full_conversation(), self.conversation_dir/fname)

    def add_persona(self, name, summary=None, img_path=None, gender=None,
                    is_custom=False, return_data=False):
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
            When True, return tuple of summary, image path, gender. Otherwise
            return None.
        """
        if (summary or img_path or gender) and not is_custom:
            raise ValueError('Can only pass in summary/img_path/gender for '
                             'custom persona.')

        processed_name = self.process_name(name)
        dir_ = self.persona_dir[is_custom]/processed_name
        if dir_.is_dir():
            if summary or img_path or gender:
                raise ValueError(
                    'Do not pass in summary/img_path/gender for a persona '
                    'that already exists.'
                )
            summary, img_path, gender = self.update_persona_dicts(
                processed_name, return_values=True, is_custom=is_custom
            )
        else:
            if is_custom:
                if not (summary and gender):
                    raise ValueError(
                        'Must provide a summary and gender for a custom '
                        'persona that does not yet exist locally.'
                    )
            else:
                summary, _, img_path, gender = wiki_data(name, img_dir=dir_,
                                                         fname='profile')

            save(summary, dir_/'summary.txt')
            save(gender, dir_/'gender.json')

            # In custom mode, we always need to move an image (either the
            # backup image or a user-specified img_path from another dir -
            # remember this is the case where no persona dir exists yet). In
            # non-custom mode, we only need to move an image if we failed to
            # download one and revert to the backup. Be careful with logic:
            # Path('abc') != 'abc', and if we convert img_path to a Path
            # immediately, we'd interpret Path('') as truthy.
            src_path = img_path or self.backup_image
            img_path = dir_/f'profile{Path(src_path).suffix}'
            try:
                if str(src_path) != str(img_path):
                    shutil.copy2(src_path, img_path)
            except FileNotFoundError as e:
                # Clean up newly-added dir otherwise this will affect
                # subsequent attempts to run this method.
                shutil.rmtree(dir_)
                raise e

            # It's an empty string if we fail to download an image in
            # non-custom mode, or None if we choose not to pass in a path in
            # custom mode.
            self.update_persona_dicts(processed_name, is_custom=is_custom)
        if return_data: return summary, img_path, gender

    def update_persona_dicts(self, processed_name, return_values=False,
                             is_custom=False):
        """Helper to update our various name2{something} dicts."""
        dir_ = self.persona_dir[is_custom]/processed_name
        summary = load(dir_/'summary.txt')
        self.name2gender[processed_name] = load(dir_/'gender.json',
                                                verbose=self.verbose)
        self.name2img_path[processed_name] = [p for p in dir_.iterdir()
                                              if p.stem == 'profile'][0]
        self.name2base[processed_name] = self._base_prompt.format(
            name=self.process_name(processed_name, inverse=True),
            summary=summary
        )
        if return_values:
            return Results(summary=summary,
                           img_path=self.name2img_path[processed_name],
                           gender=self.name2gender[processed_name])

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
            if dir_.is_dir() and all(
                    name in [path.name for path in dir_.iterdir()]
                    for name in ('gender.json', 'summary.txt')):
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
            return name.replace('_', ' ').title()
        return name.lower().replace(' ', '_').replace('.', '')

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
        mock_func = GPTBackend._get_query_func()
        kwargs_mock = kwargs.get('mock_func', None)
        # If user doesn't pass in a mock_func explicitly, this shouldn't raise
        # an error even if our backend will automatically use one.
        if kwargs_mock and kwargs_mock != mock_func:
            raise ValueError(
                f'Encountered unexpected mock_func {kwargs_mock}. The current '
                f'backend expects a mock_func of {mock_func}. Note: you '
                'typically shouldn\'t pass in mock_func explicitly since '
                'GPTBackend handles this for you. '
                '(Technically, we do allow this due to the way PromptManager '
                'and ConversationManager implement kwargs() methods).'
            )

        kwargs = {**self._kwargs, **kwargs}
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
        kwargs['mock_func'] = mock_func

        # Note: should this return an updated prompt? Right now it looks like
        # it always returns the base one.
        if name and return_prompt:
            kwargs['prompt'] = self.name2base[self.process_name(name)]
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
        if not self.current_persona:
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

        # Update these after format_prompt() call and debug check.
        self.user_turns.append(text.strip())
        self.cached_query = ''

        # Query and return generator. This allows us to use streaming mode in
        # GUI while still updating this instance with gpt3's response.
        res = GPTBackend.query(prompt, log_path=self.log_path, **kwargs)
        if not kwargs.get('stream', False):
            self.gpt3_turns.append(res[1])
            return res
        return hooked_generator(res, self.turn_hook)

    def turn_hook(self, item, i, is_post=False):
        if is_post:
            self.gpt3_turns[-1] = self.gpt3_turns[-1].strip()
        elif i == 0:
            self.gpt3_turns.append(item)
        else:
            self.gpt3_turns[-1] += item

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
        include_trailing_name

        Returns
        -------
        str
        """
        if not self.current_persona:
            raise RuntimeError('No persona loaded. Have you started a '
                               'conversation?')
        if not do_full and not user_text:
            raise RuntimeError('user_text must be provided when '
                               'do_full=False.')

        pretty_name = self.process_name(self.current_persona, inverse=True)
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
            prompt = f'{self.name2base[self.current_persona]}\n\n' + prompt
        if not include_trailing_name:
            return prompt
        return f'{prompt}\n\n{self.process_name(self.current_persona, True)}:'

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


def load_prompt(name, prompt='', rstrip=True, verbose=True, **format_kwargs):
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
        formatter = TASK2FORMATTER.get(name)
        if formatter:
            prompt = formatter(prompt_fmt, prompt, **format_kwargs)
        else:
            prompt = prompt_fmt.format(prompt)
    else:
        prompt = prompt_fmt

    # Vim adds trailing newline, which can hurt gpt3 quality.
    if rstrip: prompt = prompt.rstrip()

    # Pyyaml seems to escape newlines (probably other special characters too
    # but this is the only one I've used here, I think. Newline chars can be
    # useful in stop terms because I often use them to distinguish between
    # different examples in a prompt.
    if 'stop' in kwargs:
        kwargs['stop'] = [x.replace('\\n', '\n') for x in kwargs['stop']]
    kwargs['prompt'] = prompt
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


TASK2FORMATTER = {'conversation': conversation_formatter}

# I figure if we're importing these functions, we'll need to authenticate.
openai_auth()