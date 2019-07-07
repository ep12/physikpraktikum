'''format:pp

table format with one line of column headers  and several lines of data
'''
from warnings import warn

from utils import parse_column_header, parse_parser_options, UNIT_SYSTEMS
from physikpraktikum.measure.measurement_series import MeasurementSeries


def read_pp(filepath, options: dict = None):
    if options is None:
        opts = {}
    else:
        opts = options.copy()
    headers = []
    ln = 0
    with open(filepath) as f:
        while True:
            l = f.readline()
            ret = []
            ln += 1
            if not l:
                break
            l = l.strip()

            if l.startswith('#'): # ignore comment lines
                continue
            if l.startswith(';'): # parse options!
                opts.update(parse_parser_options(l[1:]))
                continue
            columns = l.split('|')
            if not headers:
                headers = [parse_column_header(c) for c in columns]
                nheaders = len(headers)
                continue
            for i, c in enumerate(columns):
                if i < nheaders:
                    ret.append(headers[i](c.strip()))
                else:
                    warn(UserWarning('File %r, line %d:\nData outside of the last column: %r'
                                     % (filepath, ln, c)))
            if len(columns) < nheaders:
                for i in range(len(columns) + 1, nheaders):
                    ret.append(np.NaN)
            yield ret
    del(opts, headers, nheaders)
