from prompt_toolkit import prompt


if __name__ == '__main__':
    cmd = prompt('Me: ', multiline=False, vi_mode=True)
    print('\n\nCOMMAND:\n')
    print(cmd)
