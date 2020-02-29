from functools import wraps


class WriteProtectedError(AttributeError):
    pass


class PermissionError(WriteProtectedError):
    pass


class FileFormatError(Exception):
    pass


class FormatError(Exception):
    pass


class FormatWarning(UserWarning):
    pass


class DimensionError(ValueError):
    pass


class InexactWarning(UserWarning):
    pass


class TypeWarning(UserWarning):
    pass


def deprecated(f, message: str):
    @wraps(f)
    def wrapped(*args, **kwargs):
        warn(DeprecationWarning(message, f.__name__, f.__qualname__, f.__module__), stacklevel=2)
        return f(*args, **kwargs)
    return wrapped
