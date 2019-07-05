from itertools import chain
from warnings import warn
from typing import Union, List, Callable
import numpy as np
from uncertainties import UFloat
from physikpraktikum.utils.characters import sup as superscript
from physikpraktikum.utils.representations import describe


MULTIPLICATION_SEP = ' '

# TODO: number - unit - products


class WriteProtectedError(AttributeError):
    pass


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


class UnitSystem:
    pass


class Measurement:
    pass



def _add_dict(a: dict, b: dict):
    tmp = a.copy()
    for k, v in b.items():
        if k in tmp:
            tmp[k] += v
        else:
            tmp[k] = v
    return tmp


def format_exponent(obj, exponent, method = str, neutral = '1'): # -> utils?
    if exponent == 0:
        return neutral
    elif exponent == 1:
        return method(obj)
    else:
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
            r =  us_1
    return {'UnitSystem': r}


class Unit:
    def __init__(self, quantity: str, name: str, symbol: str, tex: str, **kwargs):
        self.__dict__['quantity'], self.__dict__['name'] = quantity, name
        self.__dict__['symbol'], self.__dict__['tex'] = symbol, tex
        self.__dict__['_kwargs'] = kwargs

    def __setattr__(self, *args, **kwargs):
        raise WriteProtectedError('Unit %r is a read-only object!' % self)

    def __str__(self):
        return self.symbol

    def __repr__(self):
        return 'Unit(%r, %r, %r, %r)' % (self.quantity, self.name, self.symbol, self.tex)

    def __describe__(self):
        return '%s (symbol %r, %s)' % (self.name, self.symbol, self.quantity)

    def __mul__(self, other):
        if isinstance(other, (Unit, UnitComposition)):
            return UnitComposition(self, other, **get_unit_system(self, 1))
        if isinstance(other, Measurement):
            return Measurement(other.value, other.unit * self)
        if other == 1:
            return self
        return Measurement(other, self)

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        if isinstance(other, (Unit, UnitComposition)):
            return UnitComposition(self, other ** (-1), **get_unit_system(self, 1))
        if isinstance(other, Measurement):
            return Measurement(other.value, self / other.unit)
        return Measurement(1 / other, self)

    def __rtruediv__(self, other):
        if other == 1:
            return self ** (-1)
        if isinstance(other, (Unit, UnitComposition)):
            return UnitComposition(self ** (-1), other, **get_unit_system(self, 1))
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

    #def __bool__(self, e):
    #    return 'neutral_unit' not in self._kwargs

    #@property
    #def is_neutral(self):
    #    return 'neutral_unit' in self._kwargs


class UnitComposition:
    def __init__(self, *args, **kwargs):
        tmp = {}
        unit_systems = []
        #symbols = {}
        for a in args:
            #print(a, type(a))
            if isinstance(a, (Unit, UnitComposition)):
                us = a.get_unit_system()
                if us is not None:
                    unit_systems.append(us)
            if isinstance(a, self.__class__):
                #if str(a) in symbols:
                #    symbols[str(a)] += 1
                #else:
                #    symbols[str(a)] = 1
                tmp = _add_dict(tmp, a._dict)
            elif isinstance(a, dict): # and ...?
                tmp = _add_dict(tmp, a)
            elif isinstance(a, Unit):
                #if str(a) in symbols:
                #    symbols[str(a)] += 1
                #else:
                #    symbols[str(a)] = 1
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

    @property
    def as_base_units(self):
        if bool(self):
            # nicely sorted base units (high exponent .. low exponent)
            return MULTIPLICATION_SEP.join(format_exponent(k, v, str)
                                           for k, v in sorted(self._dict.items(), key=lambda x: x[1], reverse=True) if v != 0)
        return ''

    def __str__ (self):
        # TODO: use UnitSystem if possible
        if bool(self):
            # HOOK
            s = self._kwargs.get('NamedUnitComposition', {}).get('symbol')
            if s:
                return s
            try:
                return self.unit_system.str_find_named_derived_unit(self)
            except AttributeError:
                pass
            #except RecursionError: #HACK
            #    pass
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

    def __mul__(self, other):
        if self.is_neutral:
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
        return UnitComposition(*({k: v * e} for k, v in self._dict.items()), **get_unit_system(self, 1))

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
        return not bool(self)

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
    def tex(self):
        return self._kwargs.get('NamedUnitComposition', {}).get('tex', '')


class PoorUnit: # TODO
    '''A class to represent stupid established units (miles, hours, degree, celsius, fahrenheit)'''
    def __init__(self, name: str, symbol: str, tex: str,
                 related_base_unit: Unit,
                 to_base_unit,
                 from_base_unit):
        assert callable(to_base_unit), 'to_base_unit must be callable'
        assert callable(from_base_unit), 'from_base_unit must be callable'
        self.name, self.symbol, self.tex = name, symbol, tex
        self.base_unit = related_base_unit
        self.to_base = to_base_unit
        self.from_base = from_base_unit

    def __mul__(self, value):
        return self.to_base(value) * self.base_unit

    def __rmul__(self, value):
        return self.to_base(value) * self.base_unit

    def __truediv__(self, value):
        return self.to_base(1 / value) * self.base_unit

    def __rtruediv__(self, value):
        return 1 / self.to_base(1 / value) / self.base_unit

    def __str__(self):
        return 'PoorUnit:\t %s (symbol %r, base %s)' % (self.name, self.symbol, self.base_unit)


class UnitPrefix:
    def __init__(self, name: str, symbol: str, tex: str, factor: float):
        self.name, self.symbol, self.tex = name, symbol, tex
        self.factor = factor

    def __str__(self):
        return self.symbol

    def __repr__(self):
        return 'UnitPrefix(%r, %r, %r, %r)' % (self.name, self.symbol, self.tex, self.factor)

    def __mul__(self, other):
        return self.factor * other

    def __rmul__(self, other):
        return self.factor * other


class UnitSystem:
    def __init__(self, system_name: str):
        self.system_name = system_name
        self.base_units = {}
        self._base_units_names = []
        self._base_units_vectors = []
        self.derived_units = {}
        self.derived_units_vectors = {}
        self.aliases = {}
        self.prefixes = {}
        self.compatible_systems = []
        self.poor_units = []

    def __call__(self, quantity: str, name: str, symbol: str, tex: str,
                 as_composition: UnitComposition = None,
                 *args, **kwargs):
        if isinstance(as_composition, UnitComposition):
            as_composition._kwargs['NamedUnitComposition'] = {'quantity': quantity, 'name': name,
                                                              'symbol': symbol, 'tex': tex}
            ret = as_composition
            self.derived_units[name] = ret
            d = ret._dict # TODO: BUG
            #for k in d.keys():
            #    if k not in self.base_units:
            #        print(k)
            #        self._add_base_unit(k)
            # BUG
            v = self._calculate_vector_representation(ret)
            self.derived_units_vectors[name] = v
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
                      from_base: Callable[[float], float]):
        x = PoorUnit(name, symbol, tex, base_unit, to_base, from_base)
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
            if not unit in self.base_units.values():
                raise UnitSystemError('Unit %r is not part of the unit system %r'
                                      % (unit, self.system_name))
            return self._base_units_vectors[self._get_base_unit_index_by_unit(unit)]
        elif isinstance(unit, UnitComposition):
            d = unit._dict
            return sum(v * self._base_units_vectors[self._get_base_unit_index_by_unit(k)]
                       for k, v in d.items())
        else:
            raise TypeError('Can\'t calculate vector representation for %r' % unit)

    @property
    def units(self) -> dict:
        return {**self.base_units, **self.derived_units}

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

    @property
    def str_vector_representation(self):
        out = '<UnitSystem vectors=true name=%r>\n' % self.system_name
        for k, v in self.base_units.items():
            out += '\tBaseUnit\t: %s\t: %s (symbol %r)\n' % (self._base_units_vectors[self._base_units_names.index(k)],
                                                             v.name, v.symbol)
        for k, v in self.derived_units_vectors.items():
            u = self.derived_units[k]
            name, symbol = u.name, u.symbol
            if name and symbol:
                out += '\tUnitComposition\t: %s\t: %s (symbol %r)\n' % (v, name, symbol)
            elif name:
                out += '\tUnitComposition\t: %s\t: %s (no symbol)\n' % (v, name)
            elif symbol:
                out += '\tUnitComposition\t: %s\t: symbol %r (no long name)\n' % (v, symbol)
            else:
                out += '\tUnitComposition\t: %s\t: (no long name or symbol)\n' % v
        for v in self.prefixes.values():
            out += '\tUnitPrefix\t: %r\n' % v
        out += '</UnitSystem>'
        return out

    @property
    def as_vectors(self):
        return {**{k: v for k, v in zip(self._base_units_names, self._base_units_vectors)},
                **self.derived_units_vectors}

    def find_unit_from_string(self, unitstr: str):
        pass # TODO

    def get_neutral_unit(self):
        if self.base_units:
            u = self.base_units.values()[0]
            return u / u
        raise ValueError('Unit system %r has no units yet' % self.system_name)

    def str_find_named_derived_unit(self, unit):
        for v in chain(self.base_units.values(), self.derived_units.values()):
            if v == unit:
                return v._kwargs.get('NamedUnitComposition', {}).get('symbol', v.as_base_units)
        # TODO: Composition of multiple derived units and base units
        return unit.as_base_units

    def is_compatible(self, other_system):
        if isinstance(other_system, dict) and not other_system:
            return True
        elif other_system is None:
            return True
        return self == other_system or other in self.compatible_systems

    def check_compatible(self, other):
        if not self.is_compatible(other):
            raise UnitSystemIncompatibilityError('Incompatible unit systems %r and %r'
                                                 % (self.system_name, other.system_name))


class Measurement:
    def __init__(self, numerical_value, unit: Union[Unit, UnitComposition] = 1, unit_system: UnitSystem = None):
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
            self.__dict__['unit_system'] = unit
        else:
            self.__dict__['unit_system'] = {}

    def __setattr__(self, attr, value):
        raise WriteProtectedError('Do not manipulate data!')

    def __str__(self):
        return '%s %s' % (self.value, self.unit)

    @property
    def nominal_value(self):
        v = self.value
        if isinstance(v, UFloat):
            return v.nominal_value
        else:
            return v

    @property
    def std_dev(self):
        v = self.value
        if isinstance(v, UFloat):
            return v.std_dev
        else:
            return 0

    def is_consistent_with(self, other):
        if isinstance(other, (UFloat, self.__class__)):
            return abs(other.nominal_value - self.nominal_value) <= min(self.std_dev, other.std_dev)
        else:
            return abs(other - self.nominal_value) <= self.std_dev

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.nominal_value == other.nominal_value
        elif isinstance(other, (Unit, UnitComposition)):
            return self.unit == other
        else:
            return self.nominal_value == other

    def __mul__(self, other):
        if isinstance(other, self.__class__):
            self.unit.unit_system.check_compatible(other.unit_system)
            return Measurement(self.value * other.value, self.unit * other.unit, self.unit_system)
        elif isinstance(other, (Unit, UnitComposition)):
            return Measurement(self.value, self.unit * other, self.unit_system)
        else:
            return Measurement(self.value * other, self.unit, self.unit_system)

    def __truediv__(self, other):
        if isinstance(other, Measurement):
            self.unit.unit_system.check_compatible(other.unit_system)
            if other == 0:
                raise ZeroDivisionError(self, other)
            return Measurement(self.value / other.value, self.unit / other.unit, self.unit_system)
        elif isinstance(other, (UFloat, self.__class__)):
            if other.nominal_value == 0:
                raise ZeroDivisionError(self, other)
            return Measurement(self.value / other.value, self.unit / other.unit, self.unit_system)
        elif isinstance(other, (Unit, UnitComposition)):
            return Measurement(self.value, self.unit / other, self.unit_system)
        else:
            if other == 0:
                raise ZeroDivisionError(self, other)
            return Measurement(self.value / other, self.unit, self.unit_system)

    def __pow__(self, e):
        if self == 0:
            if e == -1:
                raise ZeroDivisionError(other, self)
            elif e == 0:
                return np.NaN
            else:
                return 1 * self.unit ** e
        return Measurement(self.value ** e, self.unit ** e, self.unit_system)

    def __rtruediv__(self, other):
        return self ** (-1) * other

    def __add__(self, other):
        if isinstance(other, Measurement):
            self.unit.unit_system.check_compatible(other.unit_system)
            if other.unit != self.unit:
                raise UnitClashError('Unit mismatch: %s and %s' % (self, other))
            return Measurement(self.value + other.value, self.unit, self.unit_system)
        else:
            if self.unit == 1 or self.unit.is_neutral:
                return other + self.value

    def __neg__(self, other):
        return Measurement(-self.value, self.unit, self.unit_system)


#def find_single_combined_unit(the_unit):
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
