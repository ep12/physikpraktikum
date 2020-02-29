import re
from warnings import warn
from collections import defaultdict

import numpy as np
from uncertainties import ufloat, ufloat_fromstr, UFloat

from physikpraktikum.errors import FormatError, InexactWarning

from physikpraktikum.measured.units.unit import PoorUnit, UnitClashError, Measurement
from physikpraktikum.measured.units.unit import Unit, UnitComposition, Measurement, PoorUnit
from physikpraktikum.measured.units.siunits import SI, angle_str_parser
from physikpraktikum.measured.measurement_series import MeasurementSeries


# NOTE: dollars not allowed as a unit
RE_COLUMN_HEADER = r'(?P<longname>.*?)? *\$(?P<symbol>[^\$]+)\$ *(in \$(?P<unit>[^\$]+)\$)? *' \
    r'((?P<relerror>rel(\.|ative)?)? *err(\.|or)? *(\+/?\-)? *(?P<errorval>\d+[,\.]?\d*(e[\+\-]?\d+)?) *' \
    r'((?P<errunit>.*)|(?P<errorpercent>%))?)?'

RE_CELL = r'(\()? *(?P<number>[^\$]+) *(?(1)\)|) *((?(1)\$?|\$) *(?P<unit>[^\$]*) *(?(1)\$?|\$))?'
# $$ mandatory if (..) not used
_CELL_NUMBER_PARSERS = [
    int, float, ufloat_fromstr,
    lambda x: ufloat_fromstr(x.replace('+-', '+/-')),
    angle_str_parser
]

RE_DECIMAL_SEPERATOR_PATTERN = r'(\d| )(,)(\d| )'
RE_DECIMAL_SEPERATOR_SUBSTITUTION = r'\1.\3'

UNIT_SYSTEMS = defaultdict(lambda: SI,
                           SI=SI)


def decompose_unit(unit): # -> unit module?
    if isinstance(unit, Measurement):
        return unit / unit.unit, unit.unit
    if isinstance(unit, (UnitComposition, Unit, PoorUnit)):
        return 1, unit
    if unit == 1:
        return 1, 1
    raise ValueError('Huh? %r? Type %r?' % (unit, type(unit)))


class ColumnHeader:
    __slots__ = ['long_name', 'symbol', 'unit', 'error', 'is_relative_error', 'options',
                 'unit_system', '_factor']
    def __init__(self, long_name: str, symbol: str, unit: str,
                 error: float, is_relative_error: bool, error_unit: str = None,
                 options: dict = None):
        if isinstance(error, (float, int)) and not error >= 0:
            error = -error
        self.long_name, self.symbol = long_name, symbol
        self.unit_system = UNIT_SYSTEMS[options.get('UnitSystem')]

        if unit == str:
            self.unit = str
        elif unit == '1':
            self.unit = 1
        elif isinstance(unit, str) and unit and unit != '1':
            unit = self.unit_system.find_unit_from_string(unit)
        if isinstance(unit, (Unit, UnitComposition, Measurement, PoorUnit)):
            self._factor, self.unit = decompose_unit(unit)
        else:
            self._factor, self.unit = 1, unit

        error_unit = (self.unit_system.find_unit_from_string(error_unit) if
                      isinstance(unit, str) and unit and unit != 1
                      else unit)
        self.is_relative_error = is_relative_error
        if isinstance(error_unit, PoorUnit): # normalise errors
            self.error = error * error_unit / error_unit.base_unit
        elif not is_relative_error:
            self.error = error * decompose_unit(error_unit)[0]  # self._factor
        else:
            self.error = error
        self.options = options if isinstance(options, dict) else {}

    @property
    def base_unit(self):
        if self.unit == 1:
            return 1
        return self.unit.base_unit

    def __str__(self):
        return '<ColumnHeader long=%r symbol=%r unit=\'%s\', err=%r rel=%r>' % (self.long_name, self.symbol,
                                                                                self.unit, self.error,
                                                                                self.is_relative_error)

    def __call__(self, raw: str): # TODO #### scaled units!!!!!!!!!!!!!!!!!
        # NOTE: WHAT HAPPENS IF:
        #  header with unit
        #  number with unit
        # TODO TESTS!
        if self.unit == str:
            return raw
        v, u = parse_data_cell(raw)
        #print('v=%r' % v, 'u=%r' % u)
        if isinstance(u, str) and u:
            u = self.unit_system.find_unit_from_string(u)
        if u is None or u == '' or u == self.unit:
            u = self.unit
            if isinstance(v, UFloat):
                v = self._factor * v * self.unit / self.base_unit
            else: # is self.error normalised?
                if self.is_relative_error:
                    v = self._factor * ufloat(v, abs(v * self.error))
                else:
                    v = ufloat(self._factor * v, self.error)
                v = v * self.unit / self.base_unit # TODO: multiply with *?
            # DONE?
        else:
            error = UnitClashError('Incompatible units: expected \'%s\', got \'%s\'' % (self.unit, u))
            # we have a unit
            # warn if we don't have a UFloat instance!
            if not isinstance(v, UFloat):
                warn(UserWarning('Are you sure that %r is exact? (raw: %r)' % (v, raw)))
            assert isinstance(u, (Unit, UnitComposition, PoorUnit)), TypeError('Expected a unit, got %r'
                                                                               % type(u))
            if isinstance(u, PoorUnit): # UNIT CLASH
                assert u.base_unit == self.unit.base_unit, error
                v = v * u / u.base_unit
            elif isinstance(u, Measurement):
                assert u.unit == self.unit.base_unit, error
                v = v * u / u.unit
            else:
                assert u == self.unit, error
        return v, self.base_unit

    def __bool__(self):
        return True

    def cadd_error(self, value):
        if not isinstance(value, UFloat) and self.error:
            if self.is_relative_error:
                return ufloat(value, self.error * value)
            else:
                return ufloat(value, self.error)
        return value

    # deprecated
    def cadd_unit(self, unit): # cond. add default unit
        if not unit:
            return self.unit
        if unit == '1':
            return 1
        return unit


def iterator_to_measurement_series(it, options):
    column_headers = next(it)  # BUG?
    columns = [MeasurementSeries(long_name=x.long_name, symbol=x.symbol, unit=x.unit,
                                 UnitSystem=x.unit_system, header=x)
               for x in column_headers]
    for x in it:
        for i, y in enumerate(x):
            n, u = y
            if u == str:
                columns[i].append(n)
            else:
                columns[i].append(n * u)
    return columns


def parse_column_header(raw_header: str, options: dict):
    #print(re.match(RE_COLUMN_HEADER, raw_header))
    m = re.fullmatch(RE_COLUMN_HEADER, raw_header.strip())
    if not m:
        # str
        return ColumnHeader(raw_header, raw_header, str, 0, False, options)
        # raise FormatError('Bad format: %r' % raw_header)
    d = m.groupdict()
    #print(d)
    long_name, symbol, unit = d['longname'] or '', d['symbol'], d['unit'] or 1
    isrel, errval, errpercent = d['relerror'], d['errorval'] or '0', d['errorpercent']
    errunit = d['errunit'] or unit
    ev = float(errval)
    if ev:
        if isrel:
            if errpercent:
                ev /= 100
                if ev > 1:
                    warn(InexactWarning('Using really bad standard error more than 100 %'))
    return ColumnHeader(long_name, symbol, unit, ev, bool(isrel), error_unit=errunit, options=options)


def parse_data_cell(raw_cell: str):
    if raw_cell.strip().upper() in ['NA', 'N/A']:
        return np.NaN, None
    raw = re.sub(RE_DECIMAL_SEPERATOR_PATTERN, RE_DECIMAL_SEPERATOR_SUBSTITUTION, raw_cell)
    m = re.fullmatch(RE_CELL, raw.strip())
    if not m:
        raise FormatError('Bad format: %r' % raw_cell)
    d = m.groupdict()
    v = d['number']
    if d['unit'] is None and not re.fullmatch('[±.,e/()\d+-]+', v):
        m2 = re.fullmatch('(?P<number>[±.,e/()\d+-]+)\s*(?P<unit>.*)', v)
        if not m2:
            raise ValueError(v, d['unit'], raw_cell, raw)
        d = m2.groupdict()
        v = d['number']
    for p in _CELL_NUMBER_PARSERS:
        try:
            v = p(v)  # sane default? last digit as error?
            break
        except Exception:
            pass
    else:
        raise FormatError('Bad format: %r\n' \
                          'See uncertainties.ufloat_fromstr documentation for supported formats'
                          % raw_cell)
    return v, d['unit']


def parse_parser_options(raw: str):
    parts = [x for x in raw.split('\t') if x]
    d = {}
    for p in parts:
        a, b = p.split('=', 1)
        a, b = a.strip(), b.strip()
        d[a] = b
    return d


if __name__ == '__main__':
    test_ch = [
        'Beschreibender ewig langer Name $x_A$ in $Meter$ rel err 5%',
        'Beschreibender ewig langer Name $x_A$ in $Meter$ err 5.7',
        '$y$', '$z$ err 5e-3', '$k$ in $Kelvin^3$', '$a$ rel err 0.1', '$a$in$Kilogram Meter Sekunde^-1$'
    ]
    test_d = [
        '(5 +- 0.1)', '8.1', '2 $Meter$', '4e2+/-1.3e-2 $Kilogram Candela^2$',
        '50 $°C$',
    ]
    print(repr(SI.find_unit_from_string('°Rø').base_unit))
    test_h = [
        ColumnHeader('Meter', '', SI.find_unit_from_string('Meter'), 0.12, False, {}),
        ColumnHeader('Inch', '', SI.find_unit_from_string('in'), 0.23, False, {}),
        ColumnHeader('Centimeter', '', SI.find_unit_from_string('cm'), 0.01, True, {}),
        ColumnHeader('Røm', '', SI.find_unit_from_string('°Rø'), 3.14, False, {})
    ]
    for x in test_ch:
        print(x)
        print(parse_column_header(x, {}))
        print()
    for x in test_d:
        print(x)
        n, u = parse_data_cell(x)
        print(n, type(n), u)
        print()
    for ch in test_h:
        print('=' * 10, 'Header %s' % ch)
        for x in test_d:
            print('Testing %r' % x)
            try:
                v, u = ch(x)
                print(v * u)
            except Exception as e:
                print(e)
            print()
        print()

