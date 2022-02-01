import os
from pathlib import Path
from prompt_toolkit import prompt, PromptSession

from jabberwocky.openai_utils import ConversationManager, PromptManager,\
    query_gpt3, query_gpt_neo, query_gpt_j
from htools.cli import fire
from htools.core import eprint, identity


os.chdir(Path('~/jabberwocky').expanduser())


def strip_and_lower(text):
    return text.strip().lower()


def prompt_until_valid(valid_responses, msg,
                       error_msg='Unrecognized input: {}',
                       postprocess=str.strip, sess=None):
    sess = sess or PromptSession(vi_mode=True)
    if not postprocess: postprocess = identity
    valid = set(valid_responses)
    while True:
        cmd = sess.prompt(msg)
        cmd = postprocess(cmd)
        if cmd in valid:
            return cmd
        else:
            print(error_msg.format(cmd))


# TODO
def talk(name, model='gptj', download_persona=False,
         engine_i=0, temperature=0.7, frequency_penalty=0.0, max_tokens=50,
         **query_kwargs):
    query_kwargs = dict(engine_i=engine_i,
                        temperature=temperature,
                        frequency_penalty=frequency_penalty,
                        max_tokens=max_tokens,
                        **query_kwargs)
    if model in ('gpt_j', 'gpt_neo'):
        query_kwargs['mock_func'] = globals()[f'query_{model}']
    conv = ConversationManager(verbose=False)
    sess = PromptSession(vi_mode=True)
    conv.start_conversation(name, download_if_necessary=download_persona)
    while True:
        query = sess.prompt('\nMe: ').strip()
        if query == '/quit':
            cmd = prompt_until_valid(
                {'y', 'n'},
                'Would you like to save this conversation? [y/n]\n',
                postprocess=strip_and_lower, sess=sess
            )
            if cmd == 'y':
                fname = sess.prompt('Enter file name to save to (not full '
                                    'path). Just press <ENTER> for default.')
                conv.end_conversation(fname=fname)
            else:
                conv.end_conversation()
                return
        else:
            print(query_kwargs)
            _, resp = conv.query(query, **query_kwargs)
            print(f'\n{conv.process_name(conv.current_persona, inverse=True)}'
                  f': {resp}')


def task():
    pass


def code():
    pass


def ls(mode=None, pretty=True):
    if not mode or mode == 'conv':
        conv = ConversationManager(verbose=False)
        print('\nAvailable Personas:')
        eprint(conv.personas(pretty=pretty))
    if not mode or mode == 'task':
        manager = PromptManager(verbose=False)
        print('\nAvailable Tasks:')
        eprint(list(manager.prompts))


if __name__ == '__main__':
    fire.Fire()
