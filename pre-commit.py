import subprocess


OPENAI_KEY_PATTERN = 'sk-[a-zA-Z0-9]{48}'


def main():
    print(__file__)
    res = subprocess.run(['ack',
                          OPENAI_KEY_PATTERN,
                          '/Users/hmamin/jabberwocky/'])
    return 1 - res.returncode


if __name__ == '__main__':
    main()


