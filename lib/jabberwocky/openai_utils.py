import numpy as np
import openai

from htools import load
from jabberwocky.config import C


def query_gpt3(prompt, engine_i=0, temperature=0.7, max_tokens=50,
               logprobs=3, mock=False, return_full=False, **kwargs):
    """Convenience function to query gpt3. If you want to stream results, use
    query_gpt3_stream() instead (any use of yield results in a generator so
    it's not straightforward to put these in 1 function. I guess the openai
    already did but I want the option of mocking results.

    Parameters
    ----------
    prompt: str
    engine_i: int
        Corresponds to engines defined in config, where 0 is the cheapest, 3 is
        the most expensive, etc.
    temperature: float
        Between 0 and 1. 0-0.4 is good for straightforward informational
        queries (e.g. reformatting, writing business emails) while 0.7-1 is
        good for more creative works.
    max_tokens: int
    logprobs: int or None
        Get log probabilities for top n candidates at each time step.
    mock: bool
        If True, return a saved sample response instead of hitting the API
        (saves tokens). Note that your other gpt3 kwargs
        (max_tokens, logprobs, kwargs) will be ignored.
        return_full will be respected since it affects the number of items
        returned - it's not a kwarg passed to the actual query function.
    return_full: bool
        If True, return a third item which is the full response object.
        Otherwise we just return the prompt and response text.
    kwargs: any
        Additional kwargs to pass to gpt3.
        Ex: presence_penalty, frequency_penalty (both floats in [0, 1]).

    Returns
    -------
    tuple: First item is prompt (str). Second is text of response (str). If
    return_full is True, a third item consisting of the whole response object
    is returned as well.
    """
    if mock:
        res = load('data/misc/sample_response.pkl')
    else:
        res = openai.Completion.create(
            engine=C.engines[engine_i],
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            logprobs=logprobs,
            **kwargs
        )
    response = res.choices[0].text
    if mock: response = f'<MOCK>{response}</MOCK>'
    result = (prompt, response)
    if return_full: result = (*result, res)
    return result


def query_gpt3_stream(prompt, engine_i=0, temperature=0.7, max_tokens=50,
                      logprobs=False, mock=False, return_full=False,
                      **kwargs):
    """Generator version of query_gpt3. Parameters are the same, but return
    values are not.

    Parameters
    ----------
    prompt: str
    engine_i: int
        Corresponds to engines defined in config, where 0 is the cheapest, 3 is
        the most expensive, etc.
    temperature: float
        Between 0 and 1. 0-0.4 is good for straightforward informational
        queries (e.g. reformatting, writing business emails) while 0.7-1 is
        good for more creative works.
    max_tokens: int
    logprobs: int or None
        Get log probabilities for top n candidates at each time step.
    mock: bool
        If True, return a saved sample response instead of hitting the API
        (saves tokens). Note that your other gpt3 kwargs
        (max_tokens, logprobs, kwargs) will be ignored.
        return_full will be respected since it affects the number of items
        returned - it's not a kwarg passed to the actual query function.
    return_full: bool
        If True, yield a third item which is the full response object.
        Otherwise we just yield the prompt and response text. (We keep the
        parameter name the same as in query_gpt3 to maintain a consistent
        interface, but technically values are yielded rather than returned.)
    kwargs: any
        Additional kwargs to pass to gpt3.
        Ex: presence_penalty, frequency_penalty (both floats in [0, 1]).

    Yields
    ------
    str or tuple: First item is response text (str). If return_full is True,
    a second item consisting of the whole response object is yielded as well.
    Because we're streaming results here, we yield 1 token (I think - if not,
    a very small chunk of text) at a time rather than returning everything at
    once.
    """
    if mock:
        stream_res = load('data/misc/sample_stream_response.pkl')
    else:
        stream_res = openai.Completion.create(
            engine=C.engines[engine_i],
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            logprobs=logprobs,
            stream=True,
            **kwargs
        )
    for chunk in stream_res:
        text = chunk.choices[0].text
        yield (text, chunk) if return_full else text


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

