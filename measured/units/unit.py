from typing import Union, List
from physikpraktikum.utils.characters import sup as superscript


MULTIPLICATION_SEP = ' '

# TODO: number - unit - products


class WriteProtectedError:
    pass


def _add_dict(a: dict, b: dict):
    tmp = a.copy()
    for k, v in b.items():
        if k in tmp:
            tmp[k] += v
        else:
            tmp[k] = v
    return tmp


def format_exponent(obj, exponent, method = str, neutral = '1'):
    if exponent == 0:
        return neutral
    elif exponent == 1:
        return method(obj)
    else:
        return method(obj) + superscript(str(exponent))


class Unit:
    def __init__(self, quantity: str, name: str, symbol: str, tex: str):
        self.__dict__['quantity'], self.__dict__['name'] = quantity, name
        self.__dict__['symbol'], self.__dict__['tex'] = symbol, tex

    def __setattr__(self, *args, **kwargs):
        raise WriteProtectedError('Unit %r is a read-only object!' % self)

    def __str__(self):
        return self.symbol

    def __repr__(self):
        return 'Unit(%r, %r, %r, %r)' % (self.quantity, self.name, self.symbol, self.tex)


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

    def __str__ (self):
        if bool(self):
            # HOOK
            s = self._kwargs.get('NamedUnitComposition', {}).get('symbol')
            if s:
                return s
            return MULTIPLICATION_SEP.join(format_exponent(k, v, str) for k, v in self._dict.items() if v != 0)
        return ''

    def __repr__(self):
        return 'UnitComposition(%r, **%r)' % (self._dict, self._kwargs)

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


Unit.__mul__ = lambda self, other: [UnitComposition(self, other), self][other == 1]
Unit.__pow__ = lambda self, e: UnitComposition({self: e})
Unit.__truediv__ = lambda self, other: UnitComposition(self, other ** (-1))
Unit.__rtruediv__ = lambda self, other: UnitComposition({self: -1}, other)


def named_unit_composition(unit_product: UnitComposition,
                           quantity: str,
                           name: str,
                           symbol: str,
                           tex: str):
    assert isinstance(unit_product, UnitComposition)
    unit_product._kwargs['NamedUnitComposition'] = {'quantity': quantity, 'name': name,
                                                    'symbol': symbol, 'tex': tex}
    return unit_product


def find_single_combined_unit(the_unit):
    for known_unit in all_well_known_units:
        if known_unit == the_unit:
            return known_unit
    return the_unit


#def register_hook_function(


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
