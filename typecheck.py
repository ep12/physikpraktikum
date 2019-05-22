import functools
import inspect
import typing
import types


DISABLED = False


class ArgumentError(Exception):
    pass


class MissingArgumentError(ArgumentError):
    pass


class ArgumentTypeError(ArgumentError):
    pass


class _Checker:
    def __init__(self,
                 varspec: dict):
        # TODO: replace pos with args!!
        if not isinstance(varspec, dict):
            raise ValueError('varspec: dict expected')
        if 'function' not in varspec or not isinstance(varspec['function'], str):
            raise ValueError('key "function" (str) must be in varspec')
        if 'name' not in varspec or not isinstance(varspec['name'], str):
            raise ValueError('key "name" (str) must be in varspec')
        if 'pos' not in varspec or not isinstance(varspec['pos'], (int, type(None))):
            raise ValueError('key "pos" (int|None) must be in varspec')
        if 'annotations' not in varspec or not isinstance(varspec['annotations'], type):
            raise ValueError('key "annotations" (type) must be in varspec')
        self.varspec = varspec
        self.fname = varspec['function']
        self.name = varspec['name']
        self.pos = varspec['pos']
        self.annotations = varspec['annotations']
        if 'default' in varspec:
            self.default = varspec['default']

    def __call__(self, *args, **kwargs):
        v = self.find_value(args, kwargs)

    def find_value(self, args, kwargs):
        if self.name in kwargs:
            return
        if self.pos is None:
            if 'default' in self.varspec:
                return kwargs.get(self.name, self.default)
            elif self.name not in kwargs:
                raise MissingArgumentError('%s: missing argument %r (%s)' % (self.fname, self.name, self.annotations))
            else:
                return kwargs[self.name]
        else:
            if not len(args) >= self.pos:  # NOPE!
                raise MissingArgumentError('%s: missing argument %r (%s)' % (self.fname, self.name, self.annotations))
            return args[pos]


def check(function):
    if DISABLED:
        return function

    argspec = inspect.getfullargspec(function)
    if not argspec.annotations:
        return function

    argnames = argspec.args

    @functools.wraps(function)
    def wrapper(*args, **kwargs):


        if not argspec.annotations:
            pass

    return wrapper


def test(function):
    #@functools.wraps
    def wrapper(*args, **kwargs):
        print(args)
        print(kwargs)
        return function(*args, **kwargs)
    return wrapper


if __name__ == '__main__':
    from pprint import pprint

    def test1(arg1: str,
              arg2: typing.List[str],
              arg3,
              arg4 = 42,
              arg5: int = 24,
              *,
              kwonlyarg6: int = 73):
              #*args,
              #**kwargs):
        return arg1

    pprint(test1.__annotations__)
    pprint(test1.__name__)
    pprint(test1.__qualname__)
    fas = inspect.getfullargspec(test1)
    pprint(fas.annotations)
    pprint(type(fas.annotations['arg1']))

    pprint((lambda x: x).__name__)

    test2 = test(test1)
    test2(arg2=1, arg4=2,arg1=3,arg5=4,arg3=5)
