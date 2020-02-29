import re
from itertools import chain
from warnings import warn
from typing import Union, List, Callable

import numpy as np
from uncertainties import UFloat

from physikpraktikum.errors import WriteProtectedError, FormatError, FormatWarning, TypeWarning
from physikpraktikum.utils.characters import sup as superscript, unsup as undo_superscript
from physikpraktikum.utils.representations import describe, longstr, str_safe
from physikpraktikum.utils.llutils import is_referenced_in_func


MULTIPLICATION_SEP = ' '
HUMAN_MULTIPLICATION_SEPS = ' ⋅*'
RE_UNIT_STRING_SPLIT = r'( +|\*\*|\*|\^|\(|\)|\\)'

_ILLEGAL_UNIT_NAME_CHARS = '^*+-,.0123456789'


class UnitSystemError(ValueError):
    pass


class UnitSystemIncompatibilityError(UnitSystemError):
    pass


class UnitWarning(UserWarning):
    pass


class UnitClashError(UnitSystemError):
    pass


class Unit:
    pass


class UnitComposition:
    pass


class NeutralUnit: # TODO rad, °, $, ...
    pass


class UnitSystem:
    pass


class Measurement:
    pass


# TODO:
# def check
# check unit system compatibility
# check unit compatibility


def avoid_np(obj):
    if hasattr(obj, 'tolist'):
        return obj.tolist()
    return obj


def _add_dict(a: dict, b: dict):
    tmp = a.copy()
    for k, v in b.items():
        if k in tmp:
            tmp[k] += v
        else:
            tmp[k] = v
    return tmp


def format_exponent(obj, exponent, method=str, neutral='1'):  # -> utils?
    if exponent == 0:
        return neutral
    elif exponent == 1:
        return method(obj)
    else:
        if int(round(exponent)) == exponent:
            return method(obj) + superscript(str(int(round(exponent))))
        return method(obj) + superscript(str(exponent))


def get_unit_system(a, b):
    try:
        us_1 = a.unit_system
    except AttributeError:
        us_1 = None
    try:
        us_2 = b.unit_system
    except AttributeError:
        us_2 = None
    if us_1 is None:
        if us_2 is None:
            r = None
        else:
            r = us_2
    else:
        if us_2 is None:
            r = us_1
        else:
            us_1.check_compatible(us_2)
            r = us_1
    return {'UnitSystem': r}


def _check_var_no_numbers(value, name):
    n = undo_superscript(value)
    if any(x in n for x in _ILLEGAL_UNIT_NAME_CHARS):
        raise ValueError('%s cannot contain the following characters: %s'
                         % (name, _ILLEGAL_UNIT_NAME_CHARS))


class Unit:
    def __init__(self, quantity: str,
                 name: str,
                 symbol: str,
                 tex: str, **kwargs):
        _check_var_no_numbers(name, 'Unit name')
        _check_var_no_numbers(symbol, 'Unit symbol')
        self.__dict__['quantity'], self.__dict__['name'] = quantity, name
        self.__dict__['symbol'], self.__dict__['tex'] = symbol, tex
        self.__dict__['_kwargs'] = kwargs

    def __setattr__(self, *args, **kwargs):
        raise WriteProtectedError('Unit %r is a read-only object!' % self)

    def __str__(self):
        return self.symbol

    def __str_safe__(self, method=str):
        if method == longstr:
            return self.name
        return self.symbol

    def __str_long__(self):
        return self.name

    def __repr__(self):
        return 'Unit(%r, %r, %r, %r)' % (self.quantity, self.name, self.symbol, self.tex)

    def __describe__(self):
        return '%s (symbol %r, %s)' % (self.name, self.symbol, self.quantity)

    # @property
    def as_base_units(self, method=str):
        return method(self)

    def __add__(self, other):  # TODO
        if isinstance(other, (Unit, UnitComposition)):
            if self == other:
                return self
            raise UnitClashError('Incompatible units: %s and %s'
                                 % (self, other))
        if isinstance(other, PoorUnit) and self != other.base_unit:
            raise UnitClashError('Incompatible units: %s and %s'
                                 % (self, other))
        else:
            return self
        raise TypeError('Trying to add a unit and %r' % type(other))

    def __mul__(self, other):
        if isinstance(other, (Unit, UnitComposition)):
            return UnitComposition(self, other, **get_unit_system(self, 1))
        if isinstance(other, Measurement):
            if other.value == 1 and other.unit * self == 1:
                return 1
            return Measurement(other.value, other.unit * self)
        if isinstance(other, np.ndarray):
            return Measurement(other, self)
        if other == 1:
            return self
        return Measurement(other, self)

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        if isinstance(other, (Unit, UnitComposition)):
            return UnitComposition(self, other ** (-1),
                                   **get_unit_system(self, 1))
        if isinstance(other, Measurement):
            return Measurement(other.value, self / other.unit)
        return Measurement(1 / other, self)

    def __rtruediv__(self, other):
        if other == 1:
            return self ** (-1)
        if isinstance(other, (Unit, UnitComposition)):
            return UnitComposition(self ** (-1), other,
                                   **get_unit_system(self, 1))
        if isinstance(other, Measurement):
            return Measurement(other.value, other.unit / self)
        return Measurement(other, self ** (-1))

    def __pow__(self, e):
        return UnitComposition({self: e}, **get_unit_system(self, 1))

    def get_unit_system(self):
        return self._kwargs.get('UnitSystem')

    @property
    def unit_system(self):
        us = self._kwargs.get('UnitSystem')
        if isinstance(us, UnitSystem):
            return us
        raise AttributeError('This unit has no associated unit system')

    @property
    def base_unit(self):
        return self

    @property
    def is_neutral(self):
        return False

    @property
    def is_named(self):
        return True

    # def __bool__(self, e):
    #    return 'neutral_unit' not in self._kwargs

    # @property
    # def is_neutral(self):
    #    return 'neutral_unit' in self._kwargs


class UnitComposition:
    def __init__(self, *args, **kwargs):
        tmp = {}
        unit_systems = []
        # symbols = {}
        for a in args:
            # print(a, type(a))
            if isinstance(a, (Unit, UnitComposition)):
                us = a.get_unit_system()
                if us is not None:
                    unit_systems.append(us)
            if isinstance(a, self.__class__):
                # if str(a) in symbols:
                #     symbols[str(a)] += 1
                # else:
                #     symbols[str(a)] = 1
                tmp = _add_dict(tmp, a._dict)
            elif isinstance(a, dict):  # and ...?
                tmp = _add_dict(tmp, a)
            elif isinstance(a, Unit):
                # if str(a) in symbols:
                #     symbols[str(a)] += 1
                # else:
                #     symbols[str(a)] = 1
                if a in tmp:
                    tmp[a] += 1
                else:
                    tmp[a] = 1
            elif a == 1:
                pass
            else:
                raise ValueError('Incompatible: %r' % a)
        self._dict = {k: v for k, v in tmp.items() if v != 0}
        self._kwargs = kwargs
        if unit_systems:
            if len(unit_systems) == 1:
                self._kwargs['UnitSystem'] = unit_systems[0]
            else:
                for i, us_1 in enumerate(unit_systems[:-1]):
                    for us_2 in unit_systems[i + 1:]:
                        us_1.check_compatible(us_2)
                self._kwargs['UnitSystem'] = unit_systems[0]

    def __eq__(self, other):
        if isinstance(other, Unit):
            if not len(self._dict) == 1:
                return False
            return list(self._dict.items())[0] in [(other, 1), (other, 1.0)]
        elif isinstance(other, self.__class__):
            if len(self._dict) != len(other._dict):
                return False
            for k, v in self._dict.items():
                if k not in other._dict or other._dict[k] != v:
                    return False
            return True
        elif other == 1:
            return all(e == 0 for e in self._dict.values())

    def __bool__(self):
        return any(x != 0 for x in self._dict.values())

    # @property
    def as_base_units(self, method=str):
        if bool(self):
            # nicely sorted base units (high exponent .. low exponent)
            return MULTIPLICATION_SEP.join(format_exponent(k, v, method)
                                           for k, v in sorted(self._dict.items(),
                                                              key=lambda x: x[1],
                                                              reverse=True)
                                           if v != 0)
        return ''

    def __str__(self):
        if bool(self):
            s = self._kwargs.get('NamedUnitComposition', {}).get('symbol')
            if s:
                return s
            return self.unit_system.str_find_named_derived_unit(self)
        return ''

    def __str_long__(self):
        if bool(self):
            s = self._kwargs.get('NamedUnitComposition', {}).get('name')
            if s:
                return s
            return self.unit_system.str_find_named_derived_unit(self, longstr)
        return ''

    def __str_safe__(self, method=str):
        '''recursion-safe variant of str'''
        if bool(self):
            s = (self._kwargs.get('NamedUnitComposition',
                                  {}).get(['symbol', 'name'][method == longstr]))
            if s:
                return s
            return self.as_base_units
        return ''

    def __repr__(self):
        return 'UnitComposition(%r, **%r)' % (self._dict, self._kwargs)

    def __describe__(self):
        if bool(self):
            s = self._kwargs.get('NamedUnitComposition', {})
            if s and s.get('symbol') and s.get('name') and s.get('quantity'):
                return '%s (symbol %r, %s)' % (s['name'], s['symbol'], s['quantity'])
        return '1 (1, Neutral unit)'

    def __add__(self, other):  # TODO
        if isinstance(other, (Unit, UnitComposition)):
            if self == other:
                return self
            raise UnitClashError('Incompatible units: %s and %s' % (self, other))
        if isinstance(other, PoorUnit) and self != other.base_unit:
            raise UnitClashError('Incompatible units: %s and %s' % (self, other))
        return self
        # raise TypeError('Trying to add a unit and %r' % type(other))

    def __mul__(self, other):
        if self.is_neutral:
            if self.name:
                # Degree.base_unit has not attr unit, base_unit -> RecursionError
                return other  # * self.unit
            return other
        if other == 1:
            return self
        if isinstance(other, Measurement):
            return Measurement(other.value, other.unit * self)
        if isinstance(other, (Unit, UnitComposition)):
            return UnitComposition(other, self, **get_unit_system(self, 1))
        return Measurement(other, self)

    def __rmul__(self, other):
        return self * other

    def __pow__(self, e):
        return UnitComposition(*({k: v * e} for k, v in self._dict.items()),
                               **get_unit_system(self, 1))

    def __truediv__(self, other):
        if isinstance(other, Measurement):
            if other == 0:
                raise ZeroDivisionError(self, other)
            return Measurement(1 / other.value, self / other.unit)
        return UnitComposition(self, other ** (-1), **get_unit_system(self, 1))

    def __rtruediv__(self, other):
        if self.is_neutral:
            return other
        if other == 1:
            return self.__class__(self ** (-1))
        if isinstance(other, Measurement):
            return Measurement(other.value, other.unit / self)
        return UnitComposition(self ** (-1), other, **get_unit_system(self, 1))

    def copy(self):
        return UnitComposition(self, **get_unit_system(self, 1))

    @property
    def is_neutral(self):
        return not bool(self) or len(self._dict) == 0

    @property
    def is_named(self):
        return self != 1 or bool(self.name)

    @property
    def is_writeable(self):
        if 'NamedUnitComposition' in self._kwargs:
            return False
        return True

    @property
    def name(self):
        return self._kwargs.get('NamedUnitComposition', {}).get('name', '')

    @property
    def symbol(self):
        return self._kwargs.get('NamedUnitComposition', {}).get('symbol', '')

    @property
    def quantity(self):
        return self._kwargs.get('NamedUnitComposition', {}).get('quantity', '')

    def get_unit_system(self):
        return self._kwargs.get('UnitSystem')

    @property
    def unit_system(self):
        us = self._kwargs.get('UnitSystem')
        if isinstance(us, UnitSystem):
            return us
        raise AttributeError('This unit has no associated unit system')

    @property
    def base_unit(self):
        return self

    @property
    def tex(self):
        return self._kwargs.get('NamedUnitComposition', {}).get('tex', '')


class PoorUnit:  # TODO
    '''A class to represent stupid established units (miles, hours, degree, celsius, fahrenheit)'''
    def __init__(self, name: str, symbol: str, tex: str,
                 related_base_unit: Unit,
                 to_base_unit,
                 from_base_unit,
                 is_linear: bool=False):
        assert callable(to_base_unit), 'to_base_unit must be callable'
        assert callable(from_base_unit), 'from_base_unit must be callable'
        self.name, self.symbol, self.tex = name, symbol, tex
        self.base_unit = related_base_unit
        self.to_base = to_base_unit
        self.from_base = from_base_unit
        self.is_linear = is_linear

    def __mul__(self, value):
        return self.to_base(value) * self.base_unit

    def __rmul__(self, value):
        return self.to_base(value) * self.base_unit

    def __truediv__(self, value):
        return self.to_base(1 / value) * self.base_unit

    def __rtruediv__(self, value):
        return 1 / self.to_base(1 / value) / self.base_unit

    def __pow__(self, e):
        if e == 1:
            return self
        if not self.is_linear:
            raise TypeError('%r is not linear!' % self.name)
        return Measurement(self.to_base(1) ** e, self.base_unit ** e)

    def __str__(self):
        return 'PoorUnit: %s (symbol %r, base %s)' % (self.name, self.symbol, self.base_unit) # no \t

    def __eq__(self, other):  # important!
        return self.base_unit == other

    @property
    def is_neutral(self):  # important!
        return self.base_unit.is_neutral

    @property
    def is_named(self):
        return True


class UnitPrefix:
    def __init__(self, name: str, symbol: str, tex: str, factor: float):
        self.name, self.symbol, self.tex = name, symbol, tex
        self.factor = factor

    def __str__(self):
        return self.symbol

    def __str_long__(self):
        return self.name

    def __repr__(self):
        return 'UnitPrefix(%r, %r, %r, %r)' % (self.name, self.symbol, self.tex, self.factor)

    def __mul__(self, other):
        if isinstance(other, (Unit, UnitComposition)):
            try:
                return Measurement(self.factor, other, other.unit_system)
            except AttributeError:
                return Measurement(self.factor, other)
        return avoid_np(np.multiply(self.factor, other))

    def __rmul__(self, other):
        return avoid_np(np.multiply(self.factor, other))

    def __pow__(self, e):
        return self.factor ** e

    # do not define __truediv__ because that is rarely used. In edge cases (Measurement str below) use * . **(-1)


class UnitSystem:
    def __init__(self, system_name: str, limit_combined_units: int = None):
        self.system_name = system_name
        self.base_units = {}
        self._base_units_names = []
        self._base_units_vectors = []
        self.derived_units = {}
        self._derived_units_vectors = {}
        self.aliases = {}
        self.prefixes = {}
        self.compatible_systems = []
        self.poor_units = []
        self._definitely_as_base_units = []
        self.limit_combined_units = limit_combined_units

    @property
    def units(self):
        return {**self.base_units, **self.derived_units}

    @property
    def all_units(self):
        return {**self.base_units, **self.derived_units, **{x.name: x for x in self.poor_units}}

    @property
    def all_unit_symbols(self):
        return {u.symbol: u for u in self.all_units.values()}

    def __call__(self, quantity: str, name: str, symbol: str, tex: str,
                 as_composition: UnitComposition = None,
                 *args, **kwargs):
        if isinstance(as_composition, UnitComposition):
            _check_var_no_numbers(name, 'Unit name')
            _check_var_no_numbers(symbol, 'Unit symbol')
            as_composition._kwargs['NamedUnitComposition'] = {'quantity': quantity, 'name': name,
                                                              'symbol': symbol, 'tex': tex}
            ret = as_composition
            self.derived_units[name] = ret
            d = ret._dict  # TODO: BUG
            # for k in d.keys():
            #    if k not in self.base_units:
            #        print(k)
            #        self._add_base_unit(k)
            # BUG
            v = self._calculate_vector_representation(ret)
            self._derived_units_vectors[name] = v
        else:
            ret = Unit(quantity, name, symbol, tex, UnitSystem=self, **kwargs)
            self._add_base_unit(ret)
        ret._kwargs['UnitSystem'] = self
        self._add_alias(ret)
        return ret

    def add_prefix(self, prefix):
        self.prefixes[prefix.name] = prefix
        return prefix

    def add_poor_unit(self, name: str, symbol: str, tex: str,
                      base_unit: Union[Unit, UnitComposition],
                      to_base: Callable[[float], float],
                      from_base: Callable[[float], float],
                      is_linear: bool = False,
                      *args, **kwargs):
        x = PoorUnit(name, symbol, tex, base_unit, to_base, from_base, is_linear, *args, **kwargs)
        self.poor_units.append(x)
        return x

    def _add_alias(self, unit):
        if isinstance(unit, Unit):
            name, symbol = unit.name, unit.symbol
        elif isinstance(unit, UnitComposition):
            kw = unit._kwargs.get('NamedUnitComposition', {})
            name, symbol = kw.get('name'), kw.get('symbol')
            if not (name and symbol):
                return
        if symbol not in self.aliases:
            self.aliases[symbol] = unit
        else:
            warn(UnitWarning('Error aliasing %r: Symbol %r is already in use (%r)'
                             % (unit, name, self.aliases[name])))

    def _add_base_unit(self, unit):
        assert isinstance(unit, Unit)
        name = unit.name
        self.base_units[name] = unit
        self._base_units_names.append(name)
        for i, v in enumerate(self._base_units_vectors):
            self._base_units_vectors[i] = np.append(v, 0)
        if self._base_units_vectors:
            l = len(self._base_units_names)
            self._base_units_vectors.append(np.array([float(i + 1 == l) for i in range(l)], dtype=float))
        else:
            self._base_units_vectors.append(np.array([1.]))

    def _get_base_unit_index_by_unit(self, unit):
        for k, v in self.base_units.items():
            if v == unit:
                return self._base_units_names.index(k)
        raise IndexError('%r is not a base unit of the unit system %r' % (unit, self.system_name))

    def _calculate_vector_representation(self, unit):
        if isinstance(unit, Unit):
            if unit not in self.base_units.values():
                raise UnitSystemError('Unit %r is not part of the unit system %r'
                                      % (unit, self.system_name))
            return self._base_units_vectors[self._get_base_unit_index_by_unit(unit)]
        elif isinstance(unit, UnitComposition):
            d = unit._dict
            t = np.zeros(len(self.base_units))
            for k, v in d.items():
                t += v * self._base_units_vectors[self._get_base_unit_index_by_unit(k)]
            return t
        else:
            raise TypeError('Can\'t calculate vector representation for %r' % unit)

    def __str__(self):
        out = '<UnitSystem name=%r>\n' % self.system_name
        for u in self.units.values():
            tag = 'BaseUnit' if isinstance(u, Unit) else 'UnitComposition'
            out += '\t%s\t: %s\n' % (tag, describe(u))
        for v in self.prefixes.values():
            out += '\tUnitPrefix\t: %r\n' % v
        for p in self.poor_units:
            out += '\t%s\n' % p
        out += '</UnitSystem>'
        return out

    def __repr__(self): # TODO
        return 'UnitSystem[%r] (no trivial repr)' % self.system_name

    def __contains__(self, other):
        return other in self.units.values()

    @property
    def str_vector_representation(self):
        out = '<UnitSystem vectors=true name=%r>\n' % self.system_name
        for k, v in self.base_units.items():
            out += ('\tBaseUnit\t: %s\t: %s (symbol %r)\n'
                    % (self._base_units_vectors[self._base_units_names.index(k)],
                       v.name, v.symbol))
        for k, v in self._derived_units_vectors.items():
            u = self.derived_units[k]
            name, symbol = u.name, u.symbol
            if name and symbol:
                out += ('\tUnitComposition\t: %s\t: %s (symbol %r)\n'
                        % (v, name, symbol))
            elif name:
                out += ('\tUnitComposition\t: %s\t: %s (no symbol)\n'
                        % (v, name))
            elif symbol:
                out += ('\tUnitComposition\t: %s\t: symbol %r (no long name)\n'
                        % (v, symbol))
            else:
                out += ('\tUnitComposition\t: %s\t: (no long name or symbol)\n'
                        % v)
        for v in self.prefixes.values():
            out += '\tUnitPrefix\t: %r\n' % v
        out += '</UnitSystem>'
        return out

    def as_vector(self, unit):
        return self._calculate_vector_representation(unit)

    @property
    def as_vectors(self):
        return {**{k: v for k, v in zip(self._base_units_names, self._base_units_vectors)},
                **self._derived_units_vectors}

    def find_unit_from_string(self, unitstr: str):
        # TODO BUG: long unit names not useable!
        # NOTE: division / not supported!
        m = re.search(RE_UNIT_STRING_SPLIT, unitstr)
        if bool(m):
            parts = [x for x in re.split(RE_UNIT_STRING_SPLIT, unitstr) if x and x != ' ']
            ret, groupstack = [], []  # [[u, u], [u, u], [u, u, u]]
            iparts = iter(parts)
            rest, e = '', None

            def reset_rest():
                nonlocal rest
                if rest:
                    warn('Ignoring %r because I don\'t '
                         'know what you mean' % rest,
                         FormatWarning)
                    rest = ''

            for p in iparts:
                unsup = undo_superscript(p)
                unchanged = ''.join(x for x, y in zip(p, unsup) if x == y)
                changed = ''.join(y for x, y in zip(p, unsup) if x != y)
                if changed:
                    try:
                        e = float(changed)
                        p = unchanged
                        # everything correct: now use that exponent
                    except Exception as err:
                        e = None
                        warn('Couldn\'t convert %r to an exponent.'
                             ' What do you want from me?\n%s'
                             % (changed, err), FormatWarning)
                # TODO: where to check for e?
                if groupstack:
                    dest = groupstack[-1]
                else:
                    dest = ret
                # print(repr(p))
                if p == '/':
                    warn('Division is not supported. Use (...)^-1 instead',
                         FormatWarning)
                    continue
                if p in ['**', '^']:  # power!
                    exponent = float(next(iparts))
                    dest[-1] = dest[-1] ** exponent
                    reset_rest()
                    continue
                if p in HUMAN_MULTIPLICATION_SEPS and p != '': # multiplication!
                    reset_rest()
                    continue
                if p == '(':  # group
                    groupstack.append([])
                    reset_rest()
                    continue
                if p == ')':  # group end
                    reset_rest()
                    if not groupstack:
                        raise FormatError('I didn\'t see that coming: \')\'')
                    if len(groupstack) == 1:
                        dest = ret
                    else:
                        dest = groupstack[-2]
                    dest.append(1)
                    for x in groupstack[-1]:
                        dest[-1] *= x
                    groupstack.pop()
                    if e:
                        dest[-1] = dest[-1] ** e
                    continue
                if p == '\\':  # latex or division
                    warn(FormatWarning('I ignore that you threw a backslash at me.\n'
                                       'If you meant a division: I don\'t support it that way, use '
                                       '(...)^-1 instead.\nIf you think I am LaTeX: No, I am not! '
                                       'You will probably encounter a ton of errors if you throw '
                                       'siunitx stuff at me! Don\'t you dare!'))
                    continue
                p = rest + p
                try:
                    if p:
                        u = self.find_unit_from_string(p)
                        dest.append(u)
                    if e:
                        dest[-1] = dest[-1] ** e
                except ValueError as e:
                    warn(FormatWarning('Error encountered, trying harder: %s'
                                       % e))
                    rest, e = p, None
            realret = 1
            # groupstack finalisation
            for x in ret:
                realret *= x
            return realret
        # TODO:
        # long names first
        if unitstr.startswith('\\'):
            if len(unitstr) > 1:
                unitstr = unitstr[1:]
            else:
                warn('Division \\ is not supported. Use (...)^-1 instead',
                     FormatWarning)
                return 1
        if '\\' in unitstr:
            i = unitstr.index('\\')
            if i == len(unitstr) - 1:
                raise FormatError('What the heck should %r mean?' % unitstr)
            return self.find_unit_from_string(unitstr[:i]) * self.find_unit_from_string(unitstr[i:])
        unsup = undo_superscript(unitstr)
        unchanged = ''.join(x for x, y in zip(unitstr, unsup) if x == y)
        changed = ''.join(y for x, y in zip(unitstr, unsup) if x != y)
        e = 1
        if changed:
            try:
                e = float(changed)
                unitstr = unchanged
            except Exception:
                raise FormatError('%r? Explaaain! Explaaain!' % unitstr)
        for n, u in self.all_units.items():
            if n == unitstr:
                return u ** e
            if u.symbol == unitstr:
                return u ** e
        for n, p in self.prefixes.items():
            if n == unitstr:
                return p ** e
            if unitstr.startswith(n):  # TODO: worx?
                return (p * self.find_unit_from_string(unitstr[len(n):])) ** e
            if p.symbol == unitstr:
                return p ** e
            if unitstr.startswith(p.symbol):
                x = unitstr[len(p.symbol):]
                for k, u in self.all_unit_symbols.items():
                    if x == k:  # TODO: worx?
                        return (p * u) ** e
        for u in self.poor_units:
            if u.name == unitstr:  # case-sensitive!
                return u ** e
            if u.symbol == unitstr:
                return u ** e
        nus = unitstr.casefold()
        for n, u in self.all_units.items():
            if n.casefold() == nus:
                return u ** e
        for n, p in self.prefixes.items():
            if n.casefold() == nus:
                return p ** e
        raise ValueError('Could not find a unit from %r' % unitstr)

    def get_neutral_unit(self):
        if self.base_units:
            u = list(self.base_units.values())[0]
            return u / u
        raise ValueError('Unit system %r has no units yet' % self.system_name)

    def _optise_vector_combination(self, unit, _dict: list = None, method=str) -> str:
        if isinstance(unit, (Unit, UnitComposition)) and unit in self._definitely_as_base_units:
            # TODO: long unit names
            return unit.as_base_units(method)
        if _dict is None:
            # TODO: long unit names
            r = self._optise_vector_combination(unit, [[], []], method=method)
            if isinstance(r, str):
                return r
            units, exponents = tuple(r)
            units = [x for _, x in sorted(zip(exponents, units),
                                          reverse=True,
                                          key=lambda t: t[0])]
            exponents = sorted(exponents, reverse=True)
            return MULTIPLICATION_SEP.join(format_exponent(str_safe(u, method),
                                                           e, method=method)
                                           for u, e in zip(units, exponents)
                                           if e != 0)
        if isinstance(unit, Unit):
            for v in self.base_units.values():
                if unit == v:
                    return [[v], [1]]
            raise ValueError('Unit %s is not part of unit system %r'
                             % (unit, self.system_name))
        if isinstance(unit, UnitComposition):
            #if len(unit._dict) == 1:
            #    return self._optise_vector_combination(list(unit._dict.keys())[0])
            v = self.as_vector(unit)
        elif isinstance(unit, np.ndarray):
            v = unit
        else:
            raise TypeError('Unsupported type %r' % type(unit))
        l = len(self.base_units)
        ul = np.count_nonzero(v)
        if (isinstance(unit, UnitComposition)
                and self.limit_combined_units
                and ul > self.limit_combined_units):
            return unit.as_base_units(method)
        gminname, gminval, gminv, gmink = None, ul, v, 0
        lpminname, lpminval, lpminv, lpmink = None, ul, v, 0
        for name, u in self.as_vectors.items(): # BUG!
            minval, minv, mink = ul, v, 0
            for index in range(l):
                if u[index] == 0 or v[index] == 0:
                    continue
                k = v[index] / u[index]
                vt = v - u * k
                n = np.count_nonzero(vt)
                if n < minval:
                    minval, minv, mink = n, vt, k
            if minval <= gminval:
                if mink != int(mink):
                    lpminname, lpminval, lpminv, lpmink = name, minval, minv, mink
                else:
                    gminname, gminval, gminv, gmink = name, minval, minv, mink
        if gminname is None:
            raise RuntimeError('This should not happen')
        # gminv has the least possible amount of non-zero items with integer k
        # if a float exponent provides a strictly better result, use that instead:
        if lpminval < gminval:
            gminname, gminval, gminv, gmink = lpminname, lpminval, lpminv, lpmink
        u = self.units[gminname]
        if u in _dict[0]:
            _dict[1][_dict[0].index(u)] += gmink
        else:
            _dict[0].append(u)
            _dict[1].append(gmink)
        if gminval:  # method=method
            return self._optise_vector_combination(gminv, _dict)
        else:
            return _dict

    def str_find_named_derived_unit(self, unit, method=str):
        for v in chain(self.base_units.values(), self.derived_units.values()):
            if v == unit:
                return str_safe(v, method)
        # TODO: Composition of multiple derived units and base units
        try:
            return self._optise_vector_combination(unit, method = method)
        except Exception:
            return str_safe(unit)

    def is_compatible(self, other_system):
        if isinstance(other_system, dict) and not other_system:
            return True
        elif other_system is None:
            return True
        return self == other_system or other_system in self.compatible_systems

    def check_compatible(self, other):
        if isinstance(other, (UnitSystem, dict)) or other is None:
            o = other
        elif isinstance(other, Measurement):
            o = other.unit_system
        else:
            raise TypeError('Incompatible type %r' % type(other))
        if not self.is_compatible(o):
            raise UnitSystemIncompatibilityError('Incompatible unit systems %r and %r'
                                                 % (self.system_name, o))


class Measurement:
    def __init__(self, numerical_value,
                 unit: Union[Unit, UnitComposition] = 1,
                 unit_system: UnitSystem = None):
        if hasattr(unit, 'unit_system') and isinstance(unit.unit_system, UnitSystem):
            fallback_unit_system = unit.unit_system
        else:
            fallback_unit_system = {}
        if isinstance(numerical_value, int):
            self.__dict__['value'] = float(numerical_value)
        else:
            self.__dict__['value'] = numerical_value
        if isinstance(unit, (Unit, UnitComposition)):
            self.__dict__['unit'] = unit
        elif unit == 1:
            if isinstance(unit_system, UnitSystem):
                try:
                    self.__dict__['unit'] = unit_system.get_neutral_unit()
                except Exception:
                    self.__dict__['unit'] = 1
            else:
                self.__dict__['unit'] = 1
        else:
            raise ValueError('%r is not a valid unit' % unit)
        if isinstance(unit, UnitSystem):
            self.__dict__['unit_system'] = unit
        elif isinstance(unit, dict) and not len(unit):
            if fallback_unit_system is None:
                self.__dict__['unit_system'] = unit
            else:
                self.__dict__['unit_system'] = fallback_unit_system
        else:
            if fallback_unit_system is None:
                self.__dict__['unit_system'] = {}
            else:
                self.__dict__['unit_system'] = fallback_unit_system

    def __setattr__(self, attr, value):
        raise WriteProtectedError('Do not manipulate data!')

    def __str__(self):
        if isinstance(self.value, UFloat):
            # TODO: use prefixes
            return '(%s) %s' % (self.value, self.unit)
        return '%s %s' % (self.value, self.unit)

    __repr__ = __str__  # ugly

    @property
    def nominal_value(self):
        v = self.value
        if isinstance(v, UFloat):
            return v.nominal_value
        else:
            return v

    n = nominal_value

    @property
    def std_dev(self):
        v = self.value
        if isinstance(v, UFloat):
            return v.std_dev
        else:
            return 0

    s = std_dev

    def is_consistent_with(self, other):
        if isinstance(other, (UFloat, self.__class__)):
            return (abs(other.nominal_value - self.nominal_value)
                    <= min(self.std_dev, other.std_dev))
        else:
            return abs(other - self.nominal_value) <= self.std_dev

    def __float__(self):
        return self.nominal_value

    def __eq__(self, other):
        if isinstance(other, Measurement):
            return self.nominal_value == other.nominal_value
        elif isinstance(other, (Unit, UnitComposition)):
            return self.unit == other
        else:
            return self.nominal_value == other

    def __mul__(self, other):
        if isinstance(other, Measurement):
            self.unit.unit_system.check_compatible(other.unit_system)
            if self.unit * other.unit == 1:  # avoid np return types
                return avoid_np(np.multiply(self.value, other.value))
            return Measurement(np.multiply(self.value, other.value),
                               self.unit * other.unit, self.unit_system)
        elif isinstance(other, (Unit, UnitComposition)):
            if self.unit * other == 1:
                return self.value
            return Measurement(self.value, self.unit * other, self.unit_system)
        elif is_referenced_in_func((other, '__mul__'), Measurement):
            return other * self
        elif is_referenced_in_func((other, '__rmul__'), Measurement):
            return NotImplemented
        elif isinstance(other, (int, float, UFloat)):
            return Measurement(np.multiply(self.value, other),
                               self.unit, self.unit_system)
        else:
            # TODO
            warn(TypeWarning('I do not know how to multiply with a %r'
                             % type(other)))
            return Measurement(np.multiply(self.value, other),
                               self.unit, self.unit_system)

    def __rmul__(self, other):
        return self * other

    def __matmul__(self, other):
        # always use numpy ufuncs for mul, div, too?
        if isinstance(other, Measurement):
            self.unit.unit_system.check_compatible(other)
            if self.unit * other.unit == 1:  # avoid np return types
                return avoid_np(np.matmul(self.value, other.value))
            return Measurement(np.matmul(self.value, other.value),
                               self.unit * other.unit, self.unit_system)
        elif is_referenced_in_func((other, '__rmatmul__'), Measurement):
            return other.__rmatmul__(self)
        else:
            return Measurement(np.matmul(self.value, other),
                               self.unit, self.unit_system)

    def __rmatmul__(self, other):
        if isinstance(other, Measurement):
            self.unit.unit_system.check_compatible(other)
            if self.unit * other.unit == 1:  # avoid np return types
                return avoid_np(np.matmul(other.value, self.value))
            return Measurement(np.matmul(other.value, self.value),
                               self.unit * other.unit, self.unit_system)
        # elif is_referenced_in_func((other, '__matmul__'), Measurement)
        # would never happen
        else:
            return Measurement(np.matmul(other, self.value),
                               self.unit, self.unit_system)

    def __truediv__(self, other):
        if isinstance(other, Measurement):
            self.unit.unit_system.check_compatible(other.unit_system)
            if other == 0:
                raise ZeroDivisionError(self, other)
            if self.unit * other.unit == 1:  # avoid np return types
                return avoid_np(np.true_divide(self.value, other.value))
            return Measurement(np.true_divide(self.value, other.value),
                               self.unit / other.unit, self.unit_system)
        elif isinstance(other, UFloat):
            if other.nominal_value == 0:
                raise ZeroDivisionError(self, other)
            return Measurement(np.true_divide(self.value, other),
                               self.unit, self.unit_system)
        elif isinstance(other, (Unit, UnitComposition)):
            if self.unit / other == 1:
                return avoid_np(self.value) # avoid np return types
            return Measurement(self.value, self.unit / other, self.unit_system)
        elif is_referenced_in_func((other, '__rtruediv__'), Measurement):
            return other.__rtruediv__(self)
        else:
            if other == 0:
                raise ZeroDivisionError(self, other)
            return Measurement(np.true_divide(self.value, other),
                               self.unit, self.unit_system)

    def __pow__(self, e):
        if self == 0:
            if e == -1:
                raise ZeroDivisionError(e, self)
            elif e == 0:
                return np.NaN
            else:
                return 1 * self.unit ** e
        if e == 0:
            return 1
        return Measurement(np.power(self.value, e),
                           self.unit ** e, self.unit_system)

    def __rtruediv__(self, other):
        # TODO: is_referenced_in_func?
        return self ** (-1) * other

    def __add__(self, other):
        if isinstance(other, Measurement):
            self.unit.unit_system.check_compatible(other.unit_system)
            if other.unit != self.unit:
                raise UnitClashError('Unit mismatch: %s and %s' % (self, other))
            if self.unit == 1:  # avoid np return types
                return avoid_np(np.add(self.value, other.value))
            return Measurement(np.add(self.value, other.value),
                               self.unit, self.unit_system)
        elif is_referenced_in_func((other, '__add__'), Measurement):
            return other + self
        elif is_referenced_in_func((other, '__radd__'), Measurement):
            return NotImplemented
        elif isinstance(other, (UFloat, int, float)):
            return Measurement(np.add(self.value, other),
                               self.unit, self.unit_system)
        else:
            if self.unit == 1 or self.unit.is_neutral:  # avoid np return types
                return avoid_np(np.add(other, self.value))
            raise TypeError('I am not sure what to do (addition of a Measurement and %r)'
                            % type(other))

    def __neg__(self):
        return Measurement(np.negative(self.value),
                           self.unit, self.unit_system)

    def __sub__(self, other):
        if isinstance(other, Measurement):
            return self + (-other)
        elif is_referenced_in_func((other, '__rsub__'), Measurement):
            return NotImplemented
        else:
            return self + np.negative(other)

    def __radd__(self, other):
        return self + other

    def __rsub__(self, other):
        return (-self) + other

    def __getitem__(self, *args):
        if hasattr(self.value, '__getitem__'):
            return self.value.__getitem__(*args)
        raise TypeError('Not subscriptable')

    def __setitem__(self, *args):
        if hasattr(self.value, '__setitem__'):
            return self.value.__setitem__(*args)
        raise TypeError('Item assignment not supported')

    def __lt__(self, other):
        if isinstance(other, Measurement):
            if self.unit == other.unit:
                return self.value < other.value
            raise UnitClashError('Incompatible units: %r and %r' % (self.unit, other.unit))
        raise NotImplemented

    @property
    def norm(self):
        try:
            return np.linalg.norm(self.value) * self.unit
        except Exception:
            pass
        return self

    def cross(self, other):
        if isinstance(other, Measurement):
            return Measurement(avoid_np(np.cross(self.value, other.value)),
                               self.unit * other.unit, self.unit_system)
        elif isinstance(other, (Unit, UnitComposition)):
            raise TypeError('Use * to multiply with a unit')
        else:
            return Measurement(avoid_np(np.cross(self.value, other)),
                               self.unit, self.unit_system)


# def find_single_combined_unit(the_unit):
#    for known_unit in all_well_known_units:
#        if known_unit == the_unit:
#            return known_unit
#    return the_unit


if __name__ == '__main__':
    a = Unit('Abstract', 'Some unit a', 'a', 'a')
    b = Unit('More abstract', 'Some unit b', 'b', 'b')
    c = Unit('Even more abstract', 'Some unit c', 'c', 'c')
    print(a * b)
    print(1 / a)
    print(a / b)
    print(a ** -10)
    print((a ** 2) ** 3 * b * c ** (-1))
    print(((a * b) / c ** 2) ** 3)
    print(a == b)
    print(a == a)
    print(a * b == b * a)
    print((a * b * b) ** 2 == a ** 2 * b ** 4)
    print(a / b == b / a)
    print(c / b != b * c)
    print(bool(a / a))
    print(bool(a ** 4 / a ** 2 * 1 / a ** 2))
    try:
        a / b >= c
    except Exception as e:
        print(e)
    print('%r\n%r\n%r' % (a, b, c))
    print(5 * a)
