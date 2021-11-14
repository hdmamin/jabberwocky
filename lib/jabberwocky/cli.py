import os
from pathlib import Path

from jabberwocky.openai_utils import ConversationManager
from htools.cli import fire
from htools.core import eprint


os.chdir(Path('~/jabberwocky').expanduser())
conv = ConversationManager()


# TODO
def talk(name, model):
    query = None
    with conv.converse():
        while query != '/quit':
            query = input('\n>>> ')
            res = conv.query(query)
            print(res)
        while True:
            cmd = input('Would you like to save that conversation? [y/n]')
            if cmd not in ('y', 'n', 'Y', 'N'):
                print(f'Unrecognized command "{cmd}"')
                continue
            if cmd == 'y':
                path = input('Output file name (not full path; just press '
                             '<ENTER> for default):')
            elif cmd == 'n':
                pass
            break


def ls(pretty=True):
    eprint(conv.personas(pretty=pretty))


if __name__ == '__main__':
    fire.Fire()
