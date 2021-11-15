import os
from pathlib import Path

from jabberwocky.openai_utils import ConversationManager, PromptManager
from htools.cli import fire
from htools.core import eprint


os.chdir(Path('~/jabberwocky').expanduser())


# TODO
def talk(name, model, download=False):
	conv = ConversationManager()
    sess = PromptSession(vi_mode=True)
    conv.start_conversation(name, download_if_necessary=download)
	query = None
    while True:
        query = sess.prompt('Me: ').strip()
        print('-' * 79)
        print('COMMAND:')
        print(query)
        print('-' * 79)
        if query == '/quit':
            cmd = prompt_until_valid(
                {'y', 'n'}, 'Would you like to save this conversation? [y/n]',
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
			input_, resp = conv.query(cmd)
			print(resp)


def task():
	pass

def code():
	pass


def ls(mode=None, pretty=True):
	if not mode or mode == 'conv':
		conv = ConversationManager()
		eprint(conv.personas(pretty=pretty))
	if not mode or mode == 'task':
		manager = PromptManager()
		eprint(list(manager.prompts))


if __name__ == '__main__':
    fire.Fire()
