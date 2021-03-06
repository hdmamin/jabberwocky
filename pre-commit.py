#!/usr/bin/env python
from pathlib import Path
import subprocess
import sys


OPENAI_KEY_PATTERN = 'sk-[a-zA-Z0-9]{48}'
YELLOW = '\033[93m'
RED = '\033[91m'
END = '\033[0m'
BOLD = '\033[1m'
PROJECT_DIR = Path('~/jabberwocky/').expanduser()


def warn(text):
    print(YELLOW + BOLD + text + END)


def error(text):
    print(RED + BOLD + text + END)


def main():
    # Piping in subprocess requires str arg instead of list and shell=True.
    print('Checking committed files for openai API keys...')
    git_matches = subprocess.run(
        f'git ls-files | ack -x {OPENAI_KEY_PATTERN}',
        shell=True
    )
    # Ack returns return code of 0 when matches are found.
    code = 1 - git_matches.returncode
    if code:
        error('\nERROR: FOUND POSSIBLE EXPOSED API KEY. \nCommit aborted. '
              'Use the `--no-verify` to force commit anyway.\n')
    else:
        # Disabled checking venv dirs because it was making commits sooo slow.
        # Use absolute paths for ignore dirs - seems like because we specify
        # the main dir to search as an absolute path, ack understandably
        # interprets all paths as absolute.
        print('Checking all files for openai API keys...')
        cmd = f'ack --ignore-dir={PROJECT_DIR/"alexa/venv"} --ignore-dir='\
              f'{PROJECT_DIR/"gui/venv"} {OPENAI_KEY_PATTERN} {PROJECT_DIR}'
        all_matches = subprocess.run(cmd.split())
        if not all_matches.returncode:
            warn('\nWARNING: FOUND POSSIBLE EXPOSED API KEY. \nAllowing '
                 'commit to proceed because it doesn\'t appear to be in a '
                 'file you\'ve committed to git, but be very careful.')
    sys.exit(code)


if __name__ == '__main__':
    main()


