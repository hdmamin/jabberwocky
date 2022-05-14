def is_yes_or_no(text, strip, lower):
    if strip: text = text.strip()
    if lower: text = text.lower()
    return text in ('yes', 'no')


def indent(text, broken=False):
    text = '\n\t'.join(line for line in text.splitlines())
    if broken:
        return text
    return '\t' + text


def test_indented(text, mode):
    bools = (line.startswith('\t') for line in text.splitlines())
    if mode == 'any':
        return any(bools)
    if mode == 'all':
        return all(bools)
    raise ValueError(f'Invalid mode: {mode}.')


def swapcase(text):
    return text.swapcase()


def reattach_1(text):
    return '1. ' + text
