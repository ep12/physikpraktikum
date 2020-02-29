SUBSCRIPT = {
    '-': '₋', '+': '₊',
    '0': '₀', '1': '₁', '2': '₂', '3': '₃', '4': '₄',
    '5': '₅', '6': '₆', '7': '₇', '8': '₈', '9': '₉',
}

SUPSCRIPT = {
    '-': '⁻', '+': '⁺', ',': '⋅', '.': '⋅',
    '0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴',
    '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹',
}


def _get_key_by_value(d, value, nf = None):
    if value not in d.values():
        return nf
    for k, v in d.items():
        if v == value:
            return k


def sup(a: str):
    return ''.join(SUPSCRIPT.get(x, x) for x in a)


def sub(a: str):
    return ''.join(SUBSCRIPT.get(x, x) for x in a)


def unsup(a: str):
    return ''.join(_get_key_by_value(SUPSCRIPT, x, x) for x in a)


def unsub(a: str):
    return ''.join(_get_key_by_value(SUBSCRIPT, x, x) for x in a)


if __name__ == '__main__':
    test = [sup('-10'), 'K' + sup('-10'), ')' + sup('-3')]
    for x in test:
        print(x, unsup(x))
