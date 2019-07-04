from warnings import warn
from typing import Union, List
import numpy as np
from physikpraktikum.utils.characters import sup as superscript
from physikpraktikum.utils.representations import describe


MULTIPLICATION_SEP = ' '

# TODO: number - unit - products


class WriteProtectedError(AttributeError):
    pass


class UnitSystemError(ValueError):
    pass


class UnitWarning(UserWarning):
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


class UnitComposition:
    def __init__(self, *args, **kwargs):
        tmp = {}
        #symbols = {}
        for a in args:
            #print(a, type(a))
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
        self._dict = tmp
        self._kwargs = kwargs
        #print('--')
        #print(symbols)
        #print('--')

    def __eq__(self, other):
        if isinstance(other, Unit):
            if not len(self._dict) == 1:
                return False
            return self._dict.items()[0] in [(other, 1), (other, 1.0)]
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
            return MULTIPLICATION_SEP.join(format_exponent(k, v, str)
                                           for k, v in self._dict.items() if v != 0)
        return ''

    def __str__ (self):
        # TODO: use UnitSystem if possible
        if bool(self):
            # HOOK
            s = self._kwargs.get('NamedUnitComposition', {}).get('symbol')
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

    def __mul__(self, other):
        if other == 1:
            return self
        return self.__class__(other, self)

    def __pow__(self, e):
        return self.__class__(*({k: v * e} for k, v in self._dict.items()))

    def __truediv__(self, other):
        return self.__class__(self, other ** (-1))

    def __rtruediv__(self, other):
        if other == 1:
            return self.__class__(self ** (-1))
        else:
            return self.__class__(self ** (-1), other)

    def copy(self):
        return self.__class__(self)

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

    @property
    def tex(self):
        return self._kwargs.get('NamedUnitComposition', {}).get('tex', '')


Unit.__mul__ = lambda self, other: [UnitComposition(self, other), self][other == 1]
Unit.__pow__ = lambda self, e: UnitComposition({self: e})
Unit.__truediv__ = lambda self, other: UnitComposition(self, other ** (-1))
Unit.__rtruediv__ = lambda self, other: UnitComposition({self: -1}, other)


#def named_unit_composition(unit_product: UnitComposition,
#                           quantity: str,
#                           name: str,
#                           symbol: str,
#                           tex: str):
#    assert isinstance(unit_product, UnitComposition)
#    unit_product._kwargs['NamedUnitComposition'] = {'quantity': quantity, 'name': name,
#                                                    'symbol': symbol, 'tex': tex}
#    return unit_product


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
        self._add_alias(ret)
        return ret

    def add_prefix(self, prefix):
        self.prefixes[prefix.name] = prefix
        return prefix

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
        out += '</UnitSystem>'
        return out

    def __repr__(self):
        pass # TODO

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
