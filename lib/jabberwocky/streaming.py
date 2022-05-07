"""Functionality to support streaming gpt responses, both from backends that
natively support it like openai and those that don't. This handles things like
updating prompt_index in each streamed response dict so we can more easily tell
which prompt a token belongs to, truncating before stop words for backends that
don't (only openai supports this natively), and updating finish_reason when
we find stop words.

Note: if adding a streaming function for a new backend, make sure to register
it so it ends up in BACKEND2STREAMER (see _stream_openai_generator for an
example). Each function must accept a first param which is the result of
calling
GPT.query(prompt, stream=True)
while using the desired backend. It must also accept **kwargs. Be careful when
changing param names - they must match the stream_response() call in
jabberwocky.openai_utils.GPTBackend._query().
"""
from collections import deque

from htools.structures import Trie
from jabberwocky.utils import containerize, register


BACKEND2STREAMER = {}


class StopWordStreamer:
    """Stream results from a GPT backend that supports streaming but either
    does not support stop word truncation OR supports it but keeps the actual
    stop word in the completion. At the moment, this only describes gooseai:
    openai truncates right before the stopword which is what I want, and
    huggingface/banana/vicgalle backends don't support streaming.

    A streamer object can be used with multiple completions (n>1) but it MUST
    be used for a single prompt (i.e. not a list of prompts).
    _stream_gooseai_generator therefore must create k streamer instances for
    each query, where k is the number of strings in the input prompt.
    """

    def __init__(self, stop_words=None, tokenizer=None, trie=None,
                 max_len=None, n=1):
        """We recommend using one of the factory constructors
        (from_trie or from_stopwords). _stream_gooseai_generator uses from_trie
        to avoid having to construct k separate tries for a query with k
        prompts.

        Either specify (stop_words and tokenizer) or (trie and max_len).

        Parameters
        ----------
        stop_words: None or list[str]
        tokenizer: None or transformers.GPT2Tokenizer
        trie: None or htools.Trie
            Trie built on a list of lists of strings. Each nested list
            corresponds to 1 stopword and each string within it corresponds to
            one of its "pretty" tokens (the tokenized form fed to the model is
            the result of BPE which creates a base vocab of 256 unicode chars
            corresponding to 256 bytes (2^8). This means there are some funny
            looking characters, e.g. "\n" maps to 'Ċ'. Completions are mapped
            back to the original characters, which is what we want to deal with
            since we'll be streaming completions). If you are creating many
            streamer objects, we suggest using the StopWordStreamer.create_trie
            to create 1 trie (and get max_len) and then using the from_trie
            constructor to create all the streamers.
        max_len: None or int
            Number of tokens in longest stopword when tokenized by GPT2.
        n: int
            Number of completions we asked to generate (same as n in
            GPT.query).

        Examples
        --------
        # For a single str prompt:
        streamer = StopWordStreamer(['\n\nMe:', '\n\nBrandon Sanderson:'],
                                    gpt2_tokenizer)
        GPT.switch('gooseai')
        response = GPT.query(prompt_str, stream=True)
        for res, full in streamer.stream(response):
            print(res, full)
        """
        if stop_words and tokenizer:
            assert trie is None and max_len is None, \
                'Do not pass in trie or max_len when specifying stop_words ' \
                'and tokenizer.'
            trie, max_len = self.create_trie(stop_words, tokenizer)
        elif trie and max_len:
            assert stop_words is None and tokenizer is None, \
                'Do not pass in stop_words or tokenizer when trie and ' \
                'max_len are provided.'
        else:
            raise RuntimeError('You must either provide stop_words and '
                               'tokenizer OR trie and max_len.')

        self.trie = trie
        self.max_len = max_len
        self.nodes = [set([self.trie.head]) for _ in range(n)]
        self.q = [deque() for _ in range(n)]
        self.done = [False for _ in range(n)]
        self.n = n
        self.used = False

    @classmethod
    def from_trie(cls, trie, max_len, n=1):
        return cls(trie=trie, max_len=max_len, n=n)

    @classmethod
    def from_stopwords(cls, stop_words, tokenizer, n=1):
        return cls(stop_words=stop_words, tokenizer=tokenizer, n=n)

    @classmethod
    def create_trie(cls, stop_words, tokenizer):
        """
        Parameters
        ----------
        stop_words
        tokenizer

        Returns
        -------
        tuple[Trie, int]: First item is trie containing all
        pretty-tokenized stop_words. Second item is the length (in # of tokens)
        of the longest stop word.
        """
        tokenized = [cls._pretty_tokenize(tokenizer, word)
                     for word in stop_words]
        max_len = max(len(toks) for toks in tokenized)
        trie = Trie(tokenized)
        return trie, max_len

    @staticmethod
    def _pretty_tokenize(tokenizer, text):
        """Tokenize text but keep all tokens in 'decoded' form. E.g. simply
        calling tokenizer.tokenize('\n\nMe:') will result in the tokens
        ['Ċ', 'Ċ', 'Me', ':'], but calling _pretty_tokenize('\n\nMe:')
        will result in the tokens ['\n', '\n', 'Me', ':']. Since this class
        is used to filter gpt completions, we need to use the latter version.
        """
        return [tokenizer.decode(tok) for tok in tokenizer.encode(text)]

    def _drop_last_n(self, token, i):
        # Check if the newest token continues any previously found partial
        # stopwords for completion i. If a full stop word has been found,
        # return a positive integer with the number of tokens that stop word
        # contains (i.e. the number to drop). If we haven't encountered a
        # terminal state in our trie, we return 0 (falsy) indicating that we
        # should not stop streaming yet.
        nodes = self.nodes[i]
        new_nodes = set([self.trie.head])
        for node in nodes:
            if node.stop_state:
                return node.depth
            if token in node.edges:
                new_nodes.add(node.edges[token])
        self.nodes[i] = new_nodes
        depths = [node.depth for node in self.nodes[i] if node.stop_state]
        return max(depths or [0])

    def stream_step(self, tok, full):
        """Stream 1 step (1 token). This basically allows us to execute
        self.stream() step by step, which is useful in the case where we have
        multiple prompts and thus multiple StopWordStreamer objects (as in
        _stream_gooseai_generator()).

        Parameters
        ----------
        tok: str
        full: dict

        Yields
        ------
        tuple[str, dict]: (token, full_response), just like in all our gpt
        stream functions/generators.
        """
        # Modulo n accounts for the case where we use this on a
        # query with multiple prompts. E.g. if we have 2 prompts, one of those
        # prompts will have completions that are not zero-indexed.
        i = full.get('index', 0) % self.n
        if self.done[i]: return
        q = self.q[i]
        to_drop = self._drop_last_n(tok, i)
        if to_drop:
            # Subtract 1 because we didn't add the latest token yet.
            for j in range(to_drop - 1):
                q.pop()
            if q: q[-1][-1]['finish_reason'] = 'stop'
            while q:
                yield q.popleft()
            self.done[i] = True
            return

        # Do not use tuple because we want this to be mutable to more
        # easily edit the finish_reason if necessary.
        q.append([tok, full])
        if len(q) > self.max_len:
            yield q.popleft()

    def stream_remaining(self):
        """After all tokens have been passed to self.stream_step(), we need to
        check and see if there are any remaining items in the queue that need
        to be streamed.

        Yields
        ------
        tuple[str, dict]: (token, full_response), just like in all our gpt
        stream functions/generators.
        """
        for i, (q, done) in enumerate(zip(self.q, self.done)):
            while q and not done:
                yield q.popleft()
            # Just update this to provide a sanity check. One we finish
            # streaming from this method, all should be done.
            self.done[i] = True

    def stream(self, gen):
        """Stream response from a call like GPT.query(prompt, stream=True).
        Note that this is built to work on a single prompt (not a list of
        multiple prompts) - that's why _stream_gooseai_generator() creates
        k streamers, where k is the number of prompts.

        Parameters
        ----------
        gen: generator
            Result of calling GPT.query(my_prompt, stream=True). At the moment
            this is only used when the gooseai backend is active since the
            other backends don't require as complex processing (gooseai is the
            only one that DOES support streaming mode but truncates AFTER
            stop words).

        Yields
        ------
        tuple[str, dict]: (token, full_response), just like in all our gpt
        stream functions/generators.
        """
        if self.used:
            raise RuntimeError(
                f'This {type(self).__name__} has already been used. You '
                'should create a new instance. Stop words can vary by prompt '
                'and even by query (users can pass in additional stop words) '
                'so it doesn\'t make sense to reuse them.'
            )

        # We refactor the step into a separate method because
        # _stream_gooseai_generator() needs to create 1 StopWordStreamer
        # for each prompt in an input query, then call each one for a single
        # step at a time.
        for tok, full in gen:
            yield from self.stream_step(tok, full)
        yield from self.stream_remaining()
        self.used = True


def stream_words(text):
    """Like stream_chars but splits on spaces. Realized stream_chars was a bad
    idea because we risk giving SPEAKER turns like
    "This is over. W" and "hat are you doing next?", neither of which would be
    pronounced as intended. We yield with a space for consistency with the
    other streaming interfaces which require no further postprocessing.
    """
    for word in text.split(' '):
        yield word + ' '


def _stream_response(text, full):
    """Generator used to stream gpt completions for backends that don't
    natively support it.

    Parameters
    ----------
    text: str
    full: dict

    Returns
    --------
    >>> text, full = query_gpt_huggingface()
    >>> text
    "It is hot"
    >>> full
    {'generated_text': 'It is hot'}

    # Notice we don't change the full response - there's no consistent
    # pattern all the non-streaming backends share.
    >>> for t, f in _stream_response(text, full):
    >>>     print(repr(t), f)
    'It ' {'generated_text': 'It is hot'}
    'is ' {'generated_text': 'It is hot'}
    'hot' {'generated_text': 'It is hot'}
    """
    # Old function used zip and itertools.cycle but this inadvertently used
    # the same dict object at each step, which can be problematic if we want
    # to save the final results (all steps' finish_reason will be set to
    # 'dummy' since we were mutating the same object.
    for tok in stream_words(text):
        yield tok, dict(full)


@register('openai', BACKEND2STREAMER)
def _stream_openai_generator(gen, n=1, **kwargs):
    """Add a prompt_index key to the dict-like full response returned by the
    openai or gooseai api when using query_gpt3(). This is only used in
    streaming mode.

    kwargs are unused - just for compatibility with other stream funcs.

    Math notes: looks like openai backend effectively sets index like this:

    ```
    for i, prompt in enumerate(prompts):
        completions = get_completions(prompt, n)
        for j, resp in enumerate(completions):
            resp['index'] = i*n + j
            yield resp

    # Or, equivalently:
    index = 0
    for i, prompt in enumerate(prompts):
        completions = get_completions(prompt, n)
        for j, resp in enumerate(completions):
            resp['index'] = index
            # Important that update happens after setting dict value.
            index += 1
            yield resp
    ```

    We want to solve for i.
    index = j + i*n
    index - j = i * n
    (index - j) / n = i
    where j < n

    We don't have j but I think index // n is equivalent in this case.
    """
    for text, full in gen:
        full['prompt_index'] = full['index'] // n
        yield text, full


@register('gooseai', BACKEND2STREAMER)
def _stream_gooseai_generator(gen, stop, tokenizer, n=1, np=1, **kwargs):
    """
    Parameters
    ----------
    gen
    stop: list[str]
        List of stopwords to truncate on.
    tokenizer
    n
    np
    kwargs: any
        Unused - just for compatibility with other stream funcs.
    """
    trie, max_len = StopWordStreamer.create_trie(stop, tokenizer)
    streamers = [StopWordStreamer.from_trie(trie, max_len, n=n)
                 for _ in range(np)]
    for text, full in gen:
        full['prompt_index'] = full['index'] // n
        yield from streamers[full['prompt_index']].stream_step(text, full)

    for streamer in streamers:
        yield from streamer.stream_remaining()


# Do not register this: it's used by many backends so it's easier to use
# dict.get().
def _stream_fake_generator(response, start_i=0, prompt_i=0, **kwargs):
    texts, fulls = containerize(*response)
    for i, (text, full) in enumerate(zip(texts, fulls)):
        queue = deque()
        gen = _stream_response(
            text,
            {**full,
             'index': i + start_i,
             'prompt_index': prompt_i,
             'finish_reason': None}
        )
        done = False
        # Yield items while checking if we're at the last item so we can mark
        # it with a finish_reason. This lets us know when one completion ends.
        while True:
            try:
                tok, tok_full = next(gen)
                queue.append((tok, tok_full))
            except StopIteration:
                done = True

            while len(queue) > 1:
                tok, tok_full = queue.popleft()
                yield tok, tok_full
            if done: break
        tok, tok_full = queue.popleft()
        tok_full['finish_reason'] = 'dummy'
        yield tok, tok_full


def stream_response(response, start_i=0, prompt_i=0, backend=None, **kwargs):
    """Generator that lets us stream tokens and metadata from gpt query
    functions for backends that don't natively provide streaming. (Obviously,
    this won't prevent backends like Huggingface from having to generate the
    full response first, but it will provide a consistent interface for us to
    use once the text has been generated). Adds some additional metadata
    compared to stream_response() which allows us to use this when n
    (# of completions) is > 1.

    Parameters
    ----------
    response: tuple or generator
        First item is texts (either str or list[str]). Second item is full
        responses (either dict or list[dict]). Alternatively, if the backend
        actually supports streaming, this is a generator that yields tuples of
        (token str, full_response dict).
    start_i: int
        When using multiple prompts, GPTBackend._query_batch passes these
        values to GPTBackend._query which in turn passes them here. They are
        used to set the index and prompt_index correctly.
    prompt_i: int
    kwargs:
        If streaming an openai response, n (int; the number of completions per
        prompt) must be provided so we can recover the prompt index.
        If streaming a gooseai response, both n (same as above) and
        np (# of prompt strings we asked for completions for) must be provided,
        as well as 'stop' (list of stopword strings) and 'tokenizer'
        (transformers.GPT2Tokenizer or transformers.GPT2TokenizerFast).

    Yields
    ------
    tuple[str, dict]: First item is a single word, second is a dict. In
    addition to whatever data the dict already had, we add a key
    'finish_reason' which is set to None unless we've hit the last token in
    the completion, in which case it's set to "dummy" (you can simply check for
    a truthy value since it's None otherwise). We also add a key 'index'
    corresponding to which completion we're in (i.e. if we have two completions
    of 3 words each, we'd want to know which completion each new token belongs
    to).

    Examples
    --------
    # This is essentially what happens inside of gpt.query():
    >>> with gpt('huggingface'):
    >>>     resp = query_gpt3(prompt, n=2, max_tokens=3)
    >>>     for tok, full in stream_response(resp, fulls):
    >>>         print(tok, full)
    The {'generated_text': 'The dog barked', 'index': 0, 'finish_reason': None}
    dog {'generated_text': 'The dog barked', 'index': 0, 'finish_reason': None}
    barked {'generated_text': 'The dog barked', 'index': 0,
            'finish_reason': 'dummy'}
    See {'generated_text': 'See Spot run', 'index': 1, 'finish_reason': None}
    Spot {'generated_text': 'See Spot run', 'index': 1, 'finish_reason': None}
    run {'generated_text': 'See Spot Run', 'index': 1,
            'finish_reason': 'dummy'}
    """
    # Backends that support streaming (openai/gooseai) must be handled a bit
    # differently. We must return after yield from because otherwise this
    # generator proceeds to the next yield statement, which is supposed to only
    # occur in the real_stream=False case.

    # jabberwocky.openai_utils should almost always have been imported when
    # using this function so GPT is almost guarantted to be present. The ugly
    # globals() usage just prevents pycharm from complaining.
    backend = backend or globals()['GPT'].current()
    streamer = BACKEND2STREAMER.get(backend, _stream_fake_generator)
    yield from streamer(response, start_i=start_i, prompt_i=prompt_i, **kwargs)
