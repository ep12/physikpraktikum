import functools
import inspect
import typing
import types

DISABLED = False


class ArgumentError(Exception):
    pass


class ArgumentErrorCollection(ArgumentError):
    def __str__(self):
        if isinstance(self.args[0], list) and all(isinstance(x, ArgumentError) for x in self.args[0]):
            return '\n\t[E] ' + '\n\t[E] '.join(str(x) for x in self.args[0])
        else:
            return str(self.args)


class MissingArgumentError(ArgumentError):
    pass


class ArgumentTypeError(ArgumentError):
    def __str__(self):
        return ', '.join('%s: %s' % i for i in self.args[0].items())


def check_value(value, annotation) -> [bool, typing.Union[Exception, None], typing.Union[str, None]]: # TODO TODO TODO TODO
    if not isinstance(annotation, (type, typing._GenericAlias)):
        raise TypeError('check_value: annotation must be a type')
    anno, val = annotation, value
    if isinstance(anno, type):
        if not isinstance(val, anno):
            return False, ArgumentTypeError({'expected': anno, 'got': type(val)}), ''
    if isinstance(anno, typing._GenericAlias):
        raise NotImplementedError('check_value')
        d = anno.__dict__
        if not d:
            # Union, Any, Dict, ...
            return anno.__subclasscheck__(type(value)), None
        elif '__args__' in d and __origin__ in d:
            args = anno.__args__
            origin = anno.__origin__
            #if anno._special: # builtin stuff
            #    if not isinstance(value, typing._GenericAlias)
            #if isinstance(
            #elif hasattr(type(value), '_special') and type(value)._special:
            #    pass


class Argument:
    def __init__(self,
                 fname,
                 name: str,
                 annotations: type,
                 has_default: bool = False,
                 default: typing.Any = None,
                 is_kwonly: bool = False):
        if not isinstance(fname, str):
            raise TypeError('Argument.__init__: fname must be a string')
        if not isinstance(name, str):
            raise TypeError('Argument.__init__: name must be a string')
        if not isinstance(annotations, (type, typing._GenericAlias)):
            raise TypeError('Argument.__init__: annotations must be a type')
        if not isinstance(has_default, bool):
            raise TypeError('Argument.__init__: has_default must be a bool')
        if not isinstance(is_kwonly, bool):
            raise TypeError('Argument.__init__: is_kwonly must be a bool')
        self.name, self.fname = name, fname
        self.annotations = annotations
        self.has_default = has_default
        self.default = default
        self.is_kwonly = is_kwonly

    def __str__(self):
        return

    def __call__(self, value):
        # TODO
        raise NotImplementedError('')


class TypeChecker:
    '''TODO'''
    def __init__(self,
                 function: typing.Callable,
                 extra_checks: typing.List[typing.Callable] = None):
        if not isinstance(function, typing.Callable):
            raise TypeError('TypeChecker.__init__: function must be an typing.Callable instance')
        if extra_checks is not None:
            if not isinstance(extra_checks, list) or not all(isinstance(x, typing.Callable) for x in extra_checks):
                raise TypeError('TypeChecker.__init__: extra_checks must be a list of callables')
        self.func = function
        self.argspec = inspect.getfullargspec(function)
        self.extra_checks = extra_checks or []
        self.checkers = {}  # dict->checker
        self._init_checkers()
        ed = self._check_defaults()
        if ed:
            raise ArgumentErrorCollection(ed)

    def __str__(self):
        return 'TypeChecker(\n function=%r,\n argspec=%r,\n extra_checks=%r)' % (self.func,
                                                                                 self.argspec,
                                                                                 self.extra_checks)

    @property
    def name(self):
        if hasattr(self, '_name'):
            return self._name
        elif hasattr(self.func, '__name__'):
            self._name = self.func.__name__
        elif hasattr(self.func, '__qualname__'):
            self._name = self.func.__qualname__
        else:
            self._name = '<unnamed function at 0x%x>' % id(self)
        return self._name

    def _init_checkers(self):
        f, aspec = self.func, self.argspec
        anno = aspec.annotations
        defaults = self._default_dict
        for k in anno:
            self.checkers[k] = Argument(fname=self.name,
                                        name=k,
                                        annotations=anno[k],
                                        has_default=k in defaults,
                                        default=defaults.get(k),
                                        is_kwonly=k in aspec.kwonlyargs)

    def _check_defaults(self):
        defaults = self._default_dict
        return {k: self.checkers[k](v) for k, v in defaults.items()}

    def __call__(self, args, kwargs):
        '''Perform the check'''
        errlist = []
        actual = self._actual_values(args, kwargs, errlist)
        if errlist:
            raise ArgumentErrorCollection(errlist)

    @property
    def _default_dict(self):
        f, aspec = self.func, self.argspec
        nargs, ndefault = len(aspec.args), len(aspec.defaults)
        argoffset = nargs - ndefault
        d = {k: v for k, v in aspec.kwonlydefaults.items() or {}.items()}
        for i in range(ndefault):
            d[aspec.args[argoffset + i]] = aspec.defaults[i]
        return d

    def _actual_values(self, args, kwargs, errlist):
        f, aspec = self.func, self.argspec
        nargs, nkwoargs = len(aspec.args), len(aspec.kwonlyargs or [])
        ndefault, nkwodefault = len(aspec.defaults), len(aspec.kwonlydefaults or {})
        argoffset = nargs - ndefault
        actual_values = self._default_dict
        for k, v in kwargs.items():
            actual_values[k] = v
        tmp_args_required = [x for x in aspec.args if x not in actual_values]
        tmp_kwargs_required = [x for x in aspec.kwonlyargs if x not in actual_values]
        for x in args:
            actual_values[tmp_args_required.pop(0)] = x
        print('Actual values:')
        pprint(actual_values)
        if tmp_args_required:
            for x in tmp_args_required:
                errlist.append(MissingArgumentError('%s: argument %r is missing' % (self.name, x)))
        if tmp_kwargs_required:
            for x in tmp_kwargs_required:
                errlist.append(MissingArgumentError('%s: kw argument %r is missing' % (self.name, x)))
        return actual_values


def check(function, *, check_defaults: bool = False):
    if DISABLED:
        return function

    argspec = inspect.getfullargspec(function)
    if not argspec.annotations:
        return function

    argnames = argspec.args
    default_args = argspec.defaults or {}
    offset = len(argnames) - len(default_args)
    defaults = {argnames[i + offset]: default_args[i] for i in range(len(default_args))}
    kwonly = argspec.kwonlyargs or []
    kwonlyd = argspec.kwonlydefaults or {}

    checkers = []
    for k, v in argspec.annotations.items():
        aspec = {
            'function': function.__name__,
            'name': k,
            'args': argnames,
            'annotations': v
        }
        if k in defaults:
            aspec['default'] = defaults[k]
        if k in kwonlyd:
            aspec['default'] = kwonlyd[k]
        checkers.append(aspec)

    #@functools.wraps(function)
    def wrapper(*args, **kwargs):
        if not argspec.annotations:
            pass
    #wrapper.__annotations__ = function.__annotations__
    #wrapper.__name__ = function.__name__
    if isinstance(check_defaults, (bool, int)) and check_defaults:
        raise NotImplementedError('default check')
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

    #@check
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
    tc = TypeChecker(test1)
    tc(['arg1', ['arg2', 'arg2'], 3], {'arg5': 5})
    tc(['arg1', ['arg2', 'arg2']], {'arg5': 5})
    #test2(arg2=1, arg4=2, arg1=3, arg5=4, arg3=5)
