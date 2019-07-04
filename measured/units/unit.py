from typing import Union, List
from physikpraktikum.utils.characters import sup as superscript


MULTIPLICATION_SEP = ' '

# TODO: number - unit - products


class Unit:
    def __init__(self, quantity: str, name: str, symbol: str, tex: str):
        self.name, self.symbol, self.tex = name, symbol, tex

    def __str__(self):
        return self.symbol


class UnitPower:
    def __init__(self, unit: Unit, exponent: int = 1):
        if isinstance(unit, self.__class__):
            self.unit, self.exponent = unit.unit, unit.exponent
        else:
            self.unit, self.exponent = unit, exponent

    def __str__(self):
        return '%s%s' % (self.unit, [superscript(str(self.exponent)), ''][self.exponent in [0, 1]])

    def __eq__(self, other):
        if other == 1:
            return self.exponent == 0
        if isinstance(other, Unit):
            return self.unit == other and self.exponent == 1
        elif isinstance(other, self.__class__):
            return self.unit == other.unit and self.exponent == other.exponent
        else:
            raise TypeError('Incompatible types %r and %r' % (type(self), type(other)))

    def copy(self):
        return self.__class__(self.unit, self.exponent)


class UnitComposition:
    def __init__(self, *args: List[Union[Unit, UnitPower]]):
        tmp = []

        def append(x):
            nonlocal tmp
            if x not in tmp:
                if isinstance(x, Unit):
                    tmp.append(UnitPower(x))
                elif isinstance(x, UnitPower):
                    tmp.append(x.copy())
                else:
                    raise TypeError('%r is not a compatible type' % x)
            else:
                yi = tmp.index(x)
                y = tmp[yi]
                if isinstance(x, UnitPower): # TODO!
                    y.exponent += x.exponent
                elif isinstance(x, Unit):
                    y.exponent += 1
                else:
                    raise TypeError('Cannot add %r to %r' % (x, y))

        for a in args:
            if isinstance(a, self.__class__):
                for x in  a._list:
                    append(x)
            elif isinstance(a, (Unit, UnitPower)):
                append(a)
            elif a == 1:
                pass
            else:
                raise ValueError('Incompatible: %r' % a)
        self._list = tmp

    def __eq__(self, other):
        if isinstance(other, (Unit, UnitPower)):
            if not len(self._list) == 1:
                return False
            return self._list[0] == other
        if len(self._list) != len(other._list):
            return False
        for x in self._list:
            if x not in other._list:
                return False
        return True

    def __str__ (self):
        return MULTIPLICATION_SEP.join(str(x) for x in self._list)



Unit.__mul__ = lambda self, other: [UnitComposition(self, other), self][other == 1]
UnitPower.__mul__ = lambda self, other: [UnitComposition(self, other), self][other == 1]
UnitComposition.__mul__ = lambda self, other: [UnitComposition(self, other), self][other == 1]
Unit.__pow__ = lambda self, e: UnitPower(self, e)
UnitPower.__pow__ = lambda self, e: UnitPower(self.unit, self.exponent * e)
Unit.__truediv__ = lambda self, other: self * (other ** (-1))
Unit.__rtruediv__ = lambda self, other: (self ** (-1)) * other
UnitPower.__truediv__ = lambda self, other: UnitComposition(self, other ** -1)
UnitPower.__rtruediv__ = lambda self, other: UnitComposition(self ** -1, other)
UnitComposition.__truediv__ = lambda self, other: UnitComposition(self, other ** -1)
UnitComposition.__pow__ = lambda self, e: UnitComposition(*(x ** e for x in self._list))


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
    print((a / a)._list) # TODO
