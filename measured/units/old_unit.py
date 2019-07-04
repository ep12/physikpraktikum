from dataclasses import dataclass
from typing import List, NewType

OPTIONS = {
    '__str__': {
        'unit_unit_product_char': ' ',
        'unit_number_product_char': ' '
    },
    '__tex__': {
        'use_siunitx': True,
    }
}

Numeric = NewType('Numeric', complex)


def is_numeric(obj):
    try:
        42 + obj
        42 * obj
        obj ** 42
        42 ** obj
        -obj
    except Exception as e:
        return False
    return True


class Unit:
    pass



@dataclass
class UnitNumberProduct:
    number: Numeric
    def __init__(self,
                 number: Numeric,
                 unit: Unit):
        assert is_numeric(number), ValueError('%r is not a numeric value' % number)
        assert issubclass(type(unit), Unit) or unit == 1, ValueError('%r is not a unit' % unit)
        self.number = number
        self.unit = unit

    def __str__(self):
        return '%s %s' % (self.number, self.unit)


@dataclass
class UnitProduct(Unit):
    baseunits: List
    exponents: List[int]

    def __init__(self,
                 baseunits: List,
                 exponents: List):
        if not len(baseunits) == len(exponents):
            raise ValueError('baseunits and exponents do not have the same length')
        self.baseunits = baseunits
        self.exponents = exponents

    def __str__(self):
        return OPTIONS.get('__str__', {}).get('unit_unit_product_char', ' ').join(x)

    def __tex__(self):
        return


@dataclass
class BaseUnit(Unit):
    symbol: str
    name: str

    def __init__(self,
                 symbol: str,
                 name: str,
                 tex: str = None):
        self.symbol = symbol
        self.name = name
        if tex is None:
            self.tex = symbol

    def __eq__(self, other):
        if not isinstance(other, BaseUnit):
            return False
        return self.name == other.name and self.symbol == other.symbol

    def __lt__(self, other):
        return False

    def __gt__(self, other):
        return False

    def __mul__(self, other):
        if isinstance(self, BaseUnit):
            if isinstance(other, BaseUnit):
                return UnitProduct([self, other], [1, 1])
            elif isinstance(other, UnitProduct):
                return other.__mul__(self)
            else:
                raise NotImplementedError('BaseUnit.__mul__')
        raise NotImplementedError('BaseUnit.__mul__')

    def __pow__(self, exp):
        assert is_numeric(exp)
        return UnitProduct([self], [exp])
