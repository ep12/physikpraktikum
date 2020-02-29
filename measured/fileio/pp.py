'''format:pp

table format with one line of column headers  and several lines of data
'''
from warnings import warn

from .utils import parse_column_header, parse_parser_options, iterator_to_measurement_series, UNIT_SYSTEMS


def iread_pp(filepath, **options):
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
                headers = [parse_column_header(c.strip(), opts) for c in columns]
                nheaders = len(headers)
                yield headers
                continue
            for i, c in enumerate(columns):
                if i < nheaders:
                    x = headers[i](c.strip())
                    ret.append(x if isinstance(x, tuple) else (x, str))
                else:
                    warn(UserWarning('File %r, line %d:\nData outside of the last column: %r'
                                     % (filepath, ln, c)))
            if len(columns) < nheaders:
                for i in range(len(columns) + 1, nheaders):
                    ret.append(np.NaN)
            yield ret
    del(opts, headers, nheaders)


def read_pp(*args, **kwargs):
    return iterator_to_measurement_series(iread_pp(*args, **kwargs), kwargs)
