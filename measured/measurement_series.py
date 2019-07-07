from typing import Union
from physikpraktikum.measured.units.unit import Unit, UnitComposition


class MeasurementSeries:
    def __init__(self,
                 data: list = None,
                 unit: Union[Unit, UnitComposition] = None,
                 *args, **kwargs):
        if unit is None:
            unit = 1
        assert isinstance(unit, (Unit, UnitComposition)) or unit == 1, TypeError('Unsupported unit')
        self.data = data
        self.unit = unit
        if isinstance(unit, (Unit, UnitComposition)):
            self.unit_system = unit.get_unit_system()
        else:
            self.unit_system = None

    def __len__(self):
        return len(self.data)

    def __abs__(self):
        pass

    def __bool__(self):
        pass

    def __add__(self, other):
        pass

    def __neg__(self):
        pass

    def __sub__(self):
        pass

    def __radd__(self, other):
        return self + other

    def __rsub__(self, other):
        return (-self) + other

    def __mul__(self, other):
        pass

    def __rmul__(self, other):
        return self * other

    def __matmul__(self, other):
        pass

    def __rmatmul__(self, other):
        pass

    def __pow__(self, e):
        pass

    def __truediv__(self, other):
        pass

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
        # TODO
        return self

    def __next__(self):
        # TODO
        raise StopIteration

    def __str__(self):
        raise NotImplemented

    def __repr__(self):
        raise NotImplemented

    def __getitem(self, index):
        pass

    def __setitem__(self, index, value):
        pass

    def index(self, obj):
        pass

    def __float__(self):
        pass
