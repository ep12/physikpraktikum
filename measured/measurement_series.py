from typing import Union, Tuple
from itertools import zip_longest

import numpy as np
from uncertainties import ufloat, UFloat, unumpy as unp

from physikpraktikum.measured.units.unit import Unit, UnitComposition, Measurement, PoorUnit, UnitClashError
from physikpraktikum.errors import DimensionError


def is_normal_value(x: object):
    if isinstance(x, UFloat):
        return is_normal_value(x.nominal_value) and is_normal_value(x.std_dev)
    if isinstance(x, float):
        return x not in (float('inf'), float('-inf')) and x == x  # x != x for x is nan
    if x is None:
        return False
    return True


class MeasurementSeries:
    pass


class MeasurementSeries:
    def __init__(self,
                 long_name: str = '',
                 symbol: str = '',
                 data: list = None,
                 unit: Union[Unit, UnitComposition] = None,
                 header = None,
                 *args, **kwargs):
        self.long_name, self.symbol = long_name, symbol
        if isinstance(data, (list, np.ndarray, tuple)):
            d2 = []
            u = None
            for x in data:
                if isinstance(x, Measurement):
                    if u is not None:
                        if x.unit != u:
                            raise UnitClashError('MeasurementSeries expects same unit numbers: %s vs %s'
                                                 % (u, x.unit))
                        # else pass
                    else:
                        u = x.unit
                    d2.append(x.value)
                    continue
                if isinstance(x, MeasurementSeries):
                    raise TypeError('You should not nest MeasurementSeries')
                d2.append(x)
            self.data = d2
        elif data is None:
            self.data = []
        else:
            raise TypeError('data argument: expected a list or similar, got %r' % type(data))
        if unit is None:
            # try:
            unit = u if u is not None else 1
            # except Exception as e:
            #    print(unit)
            #    unit = 1
        # if unit == '1':
        #     unit = 1
        if not (isinstance(unit, (Unit, UnitComposition)) or unit == 1 or unit == str): # TODO
            raise TypeError('Unsupported unit: %r' % unit)
        self.unit = unit
        if isinstance(unit, (Unit, UnitComposition)):
            self.unit_system = unit.get_unit_system()
        else:
            self.unit_system = kwargs.get('UnitSystem')
        self.header = header
        self.options = kwargs

    def __len__(self):
        return len(self.data)

    def __abs__(self):
        pass

    def __bool__(self):
        pass

    def __add__(self, other):
        if isinstance(other, MeasurementSeries):
            if len(self) != len(other):
                raise DimensionError('Trying to add MeasurementSeries with different sizes: %d ≠ %d'
                                     % (len(self), len(other)))
            # TODO: checks
            return MeasurementSeries('(sum)', '%s+%s' % (self.symbol, other.symbol),
                                     [x + y for x, y in zip(self.data, other.data)],
                                     self.unit + other.unit) # other options?
        if isinstance(other, Measurement):
            return MeasurementSeries('(sum)', '%s+[%s]' % (self.symbol, other),
                                     [x + other.value for x in self.data],
                                     self.unit + other.unit) # more?
        else: # checks? we don't have a unit
            return MeasurementSeries('(sum)', '%s+[%s]' % (self.symbol, other),
                                     [x + other for x in self.data],
                                     self.unit) # more?

    def __neg__(self):
        return MeasurementSeries('(additive inverse)', '(-%s)' % self.symbol,
                                 [-x for x in self.data],
                                 self.unit) # more?

    def __sub__(self, other):
        return self + (-other)

    def __radd__(self, other):
        return self + other

    def __rsub__(self, other):
        #print(Measurement)
        return (-self).__add__(other)

    def __mul__(self, other):
        if isinstance(other, MeasurementSeries):
            if len(self) == 1:
                return self.data[0] * self.unit * other
            elif len(other) == 1:
                return self * other.data[0] * other.unit
            elif len(self) != len(other):
                raise DimensionError('Trying to multiply MeasurementSeries with different sizes: %d ≠ %d'
                                     % (len(self), len(other)))
            return MeasurementSeries('(product)', '(%s⋅%s)' % (self.symbol, other.symbol),
                                     [x * y for x, y in zip(self.data, other.data)],
                                     self.unit * other.unit) # more?
        if isinstance(other, Measurement):
            return MeasurementSeries('(product)', '(%s⋅%s)' % (other, self.symbol),
                                     [x * other.value for x in self.data],
                                     self.unit * other.unit) # more?
        else:
            return MeasurementSeries('(product)', '(%s⋅%s)' % (other, self.symbol),
                                     [x * other for x in self.data],
                                     self.unit) # more?

    def __rmul__(self, other):
        return self * other

    def __matmul__(self, other):
        pass

    def __rmatmul__(self, other):
        pass

    def __pow__(self, e):
        if e == 0:
            return MeasurementSeries('(power)', '(%s^0)' % self.symbol,
                                     [np.NaN if x == 0 else 1 for x in self.data],
                                     1) # more?
        else:
            return MeasurementSeries('(power)', '(%s^%s)' % (self.symbol, e),
                                     [x ** e for x in self.data],
                                     self.unit ** e) # more?

    def __truediv__(self, other):
        return self * other ** (-1)

    def __rtruediv__(self, other):
        pass

    def __eq__(self, other):
        pass

    def __lt__(self, other):
        pass

    def __le__(self, other):
        pass

    def __gt__(self, other):
        pass

    def __ge__(self, other):
        pass

    def __format__(self, fmt):
        pass

    def __hash__(self):
        return id(self)

    def __iter__(self):
        self._iter_index = -1
        return self

    def __next__(self):
        self._iter_index += 1
        if self._iter_index >= len(self.data):
            del(self._iter_index)
            raise StopIteration
        return self.data[self._iter_index]

    def __str__(self):
        u = self.unit if not isinstance(self.unit, PoorUnit) else self.unit.base_unit
        return '[' + ', '.join(str(x * u) for x in self.data) + ']'

    def __repr__(self):
        return str(self)

    def __getitem__(self, index):
        if isinstance(index, (int, slice)):
            return MeasurementSeries('(selection)', '(%s[%s])' % (self.symbol, index),
                                     self.data[index], self.unit) # more?

    def __setitem__(self, index, value):
        pass

    def index(self, obj):
        pass

    def __float__(self):
        pass

    def multicol_average(self, *cols: Tuple[MeasurementSeries, ...]):
        if len(cols) == 0:
            return self
        if any(c.unit != self.unit for c in cols):
            raise UnitClashError('Cannot calculate average for columns with different units')
        new = []

        for v in zip_longest(self, *cols):
            ok = tuple(filter(is_normal_value, v))
            new.append(sum(ok) / len(ok))
        return MeasurementSeries('Multicol Average', 'avg', new, self.unit)

    @property
    def mean(self):
        m_se = sum(self.data) / len(self.data)
        m, se = m_se.nominal_value, m_se.std_dev
        v = sum((x - m_se) ** 2 for x in self.data) / len(self.data)
        std = np.sqrt(v.nominal_value)
        return ufloat(m, se + std) * self.unit

    @property
    def std_dev(self):
        m_se = sum(self.data) / len(self.data)
        m, se = m_se.nominal_value, m_se.std_dev
        v = sum((x - m_se) ** 2 for x in self.data) / len(self.data)
        return np.sqrt(v.nominal_value)

    @property
    def is_virtual(self):
        return self.options.get('virtual', False)

    def append(self, item):
        if self.header.unit == str:
            self.data.append(item)
        elif isinstance(item, Measurement):
            self.data.append(self.header.cadd_error(item.value) * item.unit / self.unit)
        else:
            self.data.append(self.header.cadd_error(item))

    def decompose_to_tuple(self):
        '''Return three seperate items: a list of numbers, a list of errors and a unit'''
        if isinstance(self.unit, Measurement):
            u = self.unit.unit
        else:
            u = self.unit
        numbers, errors = [], []
        for item in self:
            if isinstance(item, (UFloat, Measurement)):
                numbers.append(item.nominal_value)
                errors.append(item.std_dev)
            else:
                numbers.append(item)
                errors.append(0)
        return np.array(numbers), np.array(errors), u
