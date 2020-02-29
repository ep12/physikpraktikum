'''FileIO manages IO ops (multi-file database'''
from os.path import splitext
from itertools import repeat
import warnings

from physikpraktikum.errors import PermissionError, FileFormatError
from physikpraktikum.utils.human_format import fformat
from physikpraktikum.utils.representations import escape_tex, latex_format
from .pp import read_pp
#from .pp2 import read_pp2
#from .excel import read_excel

READERS = {'.pp': read_pp}

class DBFile:
    def __init__(self, path: str, permissions: str = 'r',
                 *args, **options):
        self.path = path
        self.__dict__['permissions'] = permissions
        self._columns = None
        self.options = options

    def __str__(self):
        return '<DBFile path=%r mode=%r>' % (self.path, self.permissions)

    def __setattr__(self, attr, val):
        if attr == 'permissions':
            raise PermissionError('You are not allowed to do this!')
        self.__dict__[attr] = val

    def __getitem__(self, index):
        if index == '' or index is None:
            return self
        self._ensure_cached()
        i = self._find_column_index(index)
        if i == -1:
            raise KeyError('File %r has no column %r' % (self.path, index))
        return self._columns[i]

    def __setitem__(self, index, value):
        self._ensure_cached()
        pass # TODO

    def _ensure_cached(self):
        if self._columns is None:
            self._read_file()

    def _read_file(self, **options):
        ext = splitext(self.path)[1]
        if ext not in READERS:
            raise FileFormatError('Unknown file format: %r' % ext)
        r = READERS[ext]
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter('always')
            x = r(self.path, **{**self.options, **options})
        for w in caught_warnings:
            print('WARNING (file %r):' % self.path)
            print(' ', str(w.message).replace('\n', '\n  '))
            print()
        self._columns = x

    def _find_column_index(self, index):
        r = -1
        for i, c in enumerate(self._columns):
            if c.long_name.lower() == index.lower() or c.symbol == index:
                r = i
                break
        return r

    @property
    def column_names(self):
        return [x.long_name for x in self.columns]

    @property
    def column_symbols(self):
        return [x.symbol for x in self.columns]

    @property
    def columns(self):
        self._ensure_cached()
        return self._columns

    @property
    def column_headers_latex(self):
        ch = []
        for col in self.columns:
            if col.options.get('latex'):
                ch.append(col.options['latex'])
                continue
            name, symb, unit = escape_tex(col.long_name, '\\{}'), escape_tex(col.symbol, '\\{}_^'), col.unit
            if unit == str:
                ch.append(name)
                continue
            if symb:
                x = f'{name} ${symb}$'
            else:
                x = name
            if (not hasattr(self, '_no_units_in_header')
                and (unit != 1 or
                     (hasattr(unit, 'is_named') and unit.is_named))):
                x += rf' in {unit.tex or escape_tex(unit.symbol or unit.name) or str(unit)}'
            ch.append(x)
        return ch

    def output_supertabular(self, path: str,
                            alignment: list,
                            caption: str,
                            borders: list = True,
                            caption_position: str = 'table',  # top, bottom, table
                            tablefirsthead: str = None,
                            tablehead: str = None,
                            tabletail: str = None,
                            tablelasttail: str = None,
                            variant: str = 'supertabular',
                            table_newline: str = r'\tabularnewline\hline',
                            converters: list = None,
                            stringifiers: list = None,
                            end='\n'):
        if not isinstance(borders, list) or len(borders) != len(self.columns) + 1:
            borders = [True] * (len(self.columns) + 1)
        if not isinstance(alignment, list) or len(alignment) != len(self.columns):
            if isinstance(alignment, str) and len(alignment) == 1:
                alignment = len(self.columns) * [alignment]
            else:
                alignment = ['c'] * len(self.columns)
        s_borders = list(map(lambda x: '|' * bool(x), borders))
        specifier_str = s_borders[0] + ''.join(map(lambda t: t[0] + t[1], zip(alignment, s_borders[1:])))

        if tablefirsthead is None:
            tablefirsthead = rf'\hline {" & ".join(self.column_headers_latex)} \\\hline\hline'
        if tablehead is None:
            tablehead = (rf'\hline\multicolumn{{{len(self.columns)}}}{{{s_borders[0]}c{s_borders[-1]}}}'
                         rf'{{\small continued}}{table_newline}{tablefirsthead}')
        if tabletail is None:
            tabletail = (rf'\hline\multicolumn{{{len(self.columns)}}}{{{s_borders[0]}c{s_borders[-1]}}}'
                         rf'{{\small ...}}\\\hline')
        if tablelasttail is None:
            tablelasttail = ''  # r'\hline'

        if converters is None:
            converters = [lambda x: x] * len(self.columns)

        if stringifiers is None:
            stringifiers = [latex_format] * len(self.columns)

        with open(path, 'w') as f:
            f.write(rf'\tablefirsthead{{{tablefirsthead}}}{end}')
            f.write(rf'\tablehead{{{tablehead}}}{end}')
            f.write(rf'\tabletail{{{tabletail}}}{end}')
            f.write(rf'\tablelasttail{{{tablelasttail}}}{end}')
            f.write(rf'\{caption_position}caption{{{caption}}}{end}')
            f.write(rf'\begin{{{variant}}}{{{specifier_str}}}{end}')
            for line in zip(*tuple(self.columns)):
                c_data = list(map(lambda t: t[0](t[1]), zip(converters, line)))
                s_data = map(lambda t: t[0](t[1]), zip(stringifiers, c_data))
                f.write(rf'{" & ".join(s_data)} {table_newline}{end}')
            f.write(rf'\end{{{variant}}}{end}')

    def output_tabular(self, path: str,
                            alignment: list,
                            borders: list = True,
                            table_newline: str = r'\tabularnewline\hline',
                            converters: list = None,
                            stringifiers: list = None,
                            rotate_headers: bool = False,
                            end='\n'):
        if not isinstance(borders, list) or len(borders) != len(self.columns) + 1:
            borders = [True] * (len(self.columns) + 1)
        if not isinstance(alignment, list) or len(alignment) != len(self.columns):
            if isinstance(alignment, str) and len(alignment) == 1:
                alignment = len(self.columns) * [alignment]
            else:
                alignment = ['c'] * len(self.columns)
        s_borders = list(map(lambda x: '|' * bool(x), borders))
        specifier_str = s_borders[0] + ''.join(map(lambda t: t[0] + t[1], zip(alignment, s_borders[1:])))

        if converters is None:
            converters = [lambda x: x] * len(self.columns)

        if stringifiers is None:
            stringifiers = [latex_format] * len(self.columns)

        headers = (list(map(lambda x: fr'\rotatebox{{90}}{{{x}}}', self.column_headers_latex))
                   if rotate_headers
                   else self.column_headers_latex)

        with open(path, 'w') as f:
            # f.write(rf'\{caption_position}caption{{{caption}}}{end}')
            f.write(rf'\begin{{tabular}}{{{specifier_str}}}{end}')
            f.write(rf'\hline {" & ".join(headers)} \\\hline\hline{end}')
            for line in zip(*tuple(self.columns)):
                c_data = list(map(lambda t: t[0](t[1]), zip(converters, line)))
                s_data = map(lambda t: t[0](t[1]), zip(stringifiers, c_data))
                f.write(rf'{" & ".join(s_data)} {table_newline}{end}')
            f.write(rf'\end{{tabular}}{end}')
