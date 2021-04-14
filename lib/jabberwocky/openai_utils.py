import numpy as np
import openai


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

