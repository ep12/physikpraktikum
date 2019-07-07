import re
from warnings import warn
from collections import defaultdict

from uncertainties import ufloat, ufloat_fromstr

from physikpraktikum.errors import FormatError, InexactWarning

from physikpraktikum.measured.units.siunits import SI


# NOTE: dollars not allowed as a unit
RE_COLUMN_HEADER = r'(?P<longname>.*?)? *\$(?P<symbol>[^\$]+)\$ *(in \$(?P<unit>[^\$]+)\$)? *' \
    r'((?P<relerror>rel(\.|ative)?)? *err(\.|or)? *(\+/?\-)? *(?P<errorval>\d+[,\.]?\d*(e[\+\-]?\d+)?) *' \
    r'(?P<errorpercent>%)?)?'

RE_CELL = r'(\()? *(?P<number>[^\$]+) *(?(1)\)|) *(\$ *(?P<unit>[^\$]*) *\$)?'
_CELL_NUMBER_PARSERS = [int, float, ufloat_fromstr]


UNIT_SYSTEMS = defaultdict(lambda: SI,
                           SI=SI)


class ColumnHeader:
    __slots__ = ['longname', 'symbol', 'unit', 'error', 'is_relative_error', 'options', 'unit_system']
    def __init__(self, longname: str, symbol: str, unit: str,
                 error: float, is_relative_error: bool, options: dict):
        self.longname, self.symbol = longname, symbol
        self.unit_system = UNIT_SYSTEMS[options.get('UnitSystem')]
        #self.unit = us.find_unit_from_str(unit) # TODO !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        self.error, self.is_relative_error = error, is_relative_error
        self.options = options

    def __call__(self, raw: str): # TODO
        v, u = parse_data_cell(raw)
        # int, float default error
        # unit -> column unit
        u = self.unit_system.unit_from_str(u) # TODO
        return v * u


def parse_column_header(raw_header: str, options: dict):
    #print(re.match(RE_COLUMN_HEADER, raw_header))
    m = re.fullmatch(RE_COLUMN_HEADER, raw_header.strip())
    if not m:
        raise FormatError('Bad format: %r' % raw_header)
    d = m.groupdict()
    #print(d)
    longname, symbol, unit = d['longname'] or '', d['symbol'], d['unit'] or '1'
    isrel, errval, errpercent = d['relerror'], d['errorval'] or '0', d['errorpercent']
    ev = float(errval)
    if ev:
        if isrel:
            if errpercent:
                ev /= 100
                if ev > 1:
                    warn(InexactWarning('Using really bad standard error more than 100 %'))
    return ColumnHeader(longname, symbol, unit, ev, bool(isrel), options)


def parse_data_cell(raw_cell: str):
    m = re.fullmatch(RE_CELL, raw_cell.strip())
    if not m:
        raise FormatError('Bad format: %r' % raw_cell)
    d = m.groupdict()
    v = d['number']
    for p in _CELL_NUMBER_PARSERS:
        try:
            v = p(v)
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
        'Beschreibender ewig langer Name $x_A$ in $Meter$ rel err 5%', '$y$', '$z$ err 5e-3',
        '$k$ in $Kelvin^3$', '$a$ rel err 0.1', '$a$in$Kilogram Meter Sekunde^-1$'
    ]
    test_d = [
        '(5 +/- 0.1)', '8.1', '2 $Meter$', '4e2+/-1.3e-2 $Kilogram Candela^2$'
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
