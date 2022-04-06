import subprocess
import sys


OPENAI_KEY_PATTERN = 'sk-[a-zA-Z0-9]{48}'


def main():
    print('In commit hook', __file__)
    res = subprocess.run(['ack',
                          OPENAI_KEY_PATTERN,
                          '/Users/hmamin/jabberwocky/'])
    sys.exit(1 - res.returncode)


if __name__ == '__main__':
    main()


