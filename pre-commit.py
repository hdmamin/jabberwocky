#!/usr/bin/env python
from pathlib import Path
import subprocess
import sys


OPENAI_KEY_PATTERN = 'sk-[a-zA-Z0-9]{48}'


def main():
    # Piping in subprocess requires str arg instead of list and shell=True.
    git_matches = subprocess.run(
        f'git ls-files | ack -x {OPENAI_KEY_PATTERN}',
        shell=True
    )
    # Ack returns return code of 0 when matches are found.
    code = 1 - git_matches.returncode
    if code:
        print('\nERROR: FOUND POSSIBLE EXPOSED API KEY. \nCommit aborted. '
              'Use the `--no-verify` to force commit anyway.\n')
    else:
        all_matches = subprocess.run(
            ['ack', OPENAI_KEY_PATTERN, Path('~/jabberwocky/').expanduser()]
        )
        if not all_matches.returncode:
            print('\nWARNING: FOUND POSSIBLE EXPOSED API KEY. \nAllowing '
                  'commit to proceed because it doesn\'t appear to be in a '
                  'file you\'ve committed to git, but be very careful.')
    sys.exit(code)


if __name__ == '__main__':
    main()


