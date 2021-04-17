import numpy as np
import openai

from htools import load
from jabberwocky.config import C
from jabberwocky.utils import openai_auth


def query_gpt3(prompt, engine_i=0, temperature=0.7, max_tokens=50,
               logprobs=None, stream=False, mock=False, return_full=False,
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
        texts = (chunk.choices[0].text for chunk in res)
        return zip(texts, res) if return_full else texts
    else:
        output = (prompt, res.choices[0].text, res)
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


# I figure if we're importing these functions, we'll need to authenticate.
openai_auth()

