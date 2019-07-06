from warnings import warn
from functools import wraps


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


describe = _create_method('describe')
tex = _create_method('tex', warn_no_attr=True)
longstr = _create_method('str_long', fname='longstr')
str_safe = _create_method('str_safe')
