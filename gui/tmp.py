"""
Prototyping potential CLI here. Eventually integrate this into cli.py in
library if I like how this ends up.
"""

from prompt_toolkit import prompt, PromptSession

from htools.core import identity


def strip_and_lower(text):
    return text.strip().lower()


def prompt_until_valid(valid_responses, msg,
                       error_msg='Unrecognized input: {}.',
                       postprocess=str.strip, sess=None):
    sess = sess or PromptSession(vi_mode=True)
    if not postprocess: postprocess = identity
    valid = set(valid_responses)
    while True:
        cmd = sess.prompt(msg)
        cmd = postprocess(cmd)
        if cmd in valid_responses:
            return cmd
        else:
            print(error_msg.format(cmd))


def main():
    sess = PromptSession(vi_mode=True)
    while True:
        cmd = sess.prompt('Me: ').strip()
        print('-' * 79)
        print('COMMAND:')
        print(cmd)
        print('-' * 79)
        if cmd == '/quit':
            cmd = prompt_until_valid(
                {'y', 'n'}, 'Would you like to save this conversation? [y/n]',
                postprocess=strip_and_lower, sess=sess
            )
            if cmd == 'y':
                fname = sess.prompt('Enter file name to save to (not full '
                                    'path). Just press <ENTER> for default.')

            else:
                return

if __name__ == '__main__':
    main()
