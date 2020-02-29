from warnings import warn
from functools import wraps

from uncertainties import UFloat

TEX_ESCAPE = {
    '&': r'\&',
    '%': r'\%',
    '$': r'\$',
    '#': r'\#',
    '_': r'\_',
    '{': r'\{',
    '}': r'\}',
    '~': r'\textasciitilde',
    '^': r'\textasciicircum',
    '\\': r'\textbackslash',
}


class RepresentationWarning(Warning):
    pass


def _create_method(attrname: str, *,
                   warn_no_attr: bool = False, alt_method = str,
                   fname: str = None, qname: str = None):
    a = '__%s__' % attrname
    def f(obj, *args, **kwargs):
        if hasattr(obj, a):
            x = getattr(obj, a)
            if isinstance(x, str):
                return x
            return x(*args, **kwargs)
        if warn_no_attr:
            warn(RepresentationWarning('Object %r at 0x%x has no attribute %s' % (obj, id(obj), a)))
        return alt_method(obj)
    f.__name__ = fname or attrname
    f.__qualname__ = qname or fname or attrname
    f.__doc__ = 'Return a string representation (%s) of an object (attribute %r)' % (fname or attrname, a)
    return f


def escape_tex(raw, keep: list = None):
    if keep is None:
        keep = []
    for c, r in TEX_ESCAPE.items():
        if c in keep:
            continue
        raw = raw.replace(c, r)
    return raw


def latex_format(x):
    if isinstance(x, UFloat):
        if x.nominal_value != x.nominal_value:
            return 'NA'
        if x.nominal_value in [float('inf'), float('-inf')]:
            return '$' + '-' * (x.nominal_value < 0) + '\\infty$'
        return '${:.2uL}$'.format(x)
    return str(x)


describe = _create_method('describe')
tex = _create_method('tex', warn_no_attr=True)
longstr = _create_method('str_long', fname='longstr')
str_safe = _create_method('str_safe')
