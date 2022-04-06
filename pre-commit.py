#!/usr/bin/env python
from pathlib import Path
import subprocess
import sys


OPENAI_KEY_PATTERN = 'sk-[a-zA-Z0-9]{48}'


def main():
    res = subprocess.run(['ack',
                          OPENAI_KEY_PATTERN,
                          Path('~/jabberwocky/').expanduser()])
    if not res.returncode:
        print('\nWARNING: FOUND POSSIBLE EXPOSED API KEY. \nCommit aborted. '
              'Use the `--no-verify` to force commit anyway.')
    sys.exit(1 - res.returncode)


if __name__ == '__main__':
    main()


