'''PhysikPraktikum tools'''

import sys
import os
from itertools import combinations, permutations, zip_longest as zip_longest
from typing import Union, Any, Hashable, Tuple, List, List, Dict, Type, Set
from typing import NoReturn as Void, Callable as Func, Optional as Opt

# from typecheck import typecheck as _check #broken

import numpy as np
from scipy.optimize import curve_fit
import matplotlib
from matplotlib import pyplot as plt

from uncertainties import ufloat, UFloat

from physikpraktikum.measured.measurement_series import MeasurementSeries


SCATTERPLOT_OPTIONS = {
    'marker': 'x',
    'markersize': 2.5,
    'linewidth': 0,
}
FITPLOT_OPTIONS = {
    'marker': None,
    'linewidth': 1
}
ERRORBAR_OPTIONS = {
    'fmt': 'none',
    'elinewidth': 1.5,
    'alpha': 0.5,
}
COLORS = ['black', 'blue', 'red', 'green', 'magenta', 'orange', 'grey', 'cyan',
          'darkred', 'olive', 'turquoise', 'darkslategrey', 'teal', 'navy',
          'darkorchid', 'royalblue', 'crimson', 'purple', 'deepskyblue', 'lime',
          'sienna', 'goldenrod']
LINESTYLES = ['-', '--', ':', '-.']
SCISTYLE = matplotlib.RcParams(**{ # see matplotlib.pyplot.style.library['classic']
    'axes.grid': True,
    'axes.grid.axis': 'both',
    'axes.grid.which': 'both',  # major, minor, both
    'axes.spines.top': False,  # Boxplot lines on the top and right edge
    'axes.spines.right': False,
    'figure.titlesize': 18,
    'figure.subplot.left': 0.075,
    'figure.subplot.right': 0.925,
    'figure.subplot.top': 0.9,
    'figure.subplot.bottom': 0.1,
    'figure.subplot.hspace': 0.2,
    'figure.subplot.wspace': 0.2,
    'font.size': 12.0,
    'grid.linestyle': '-',
    'lines.marker': '',
    'scatter.marker': 'x',
    'legend.fontsize': 10,
    'legend.loc': 'best',
    'legend.labelspacing': 0.75,
    #'savefig.format': 'svg',
    'savefig.pad_inches': 0.025,
    'text.usetex': True,
    # 'text.latex.unicode': True,  # deprecated
    # 'pgf.texsystem': 'xelatex',
    'text.latex.preamble': [r'\usepackage{siunitx}',
                            # r'\usepackage{latexsym}',
                            # r'\usepackage{stmaryrd}',  # symbols
                            r'\usepackage{amsmath}',
                            # r'\usepackage{amssymb}',
                            # r'\usepackage{amsfonts}',
                            r'\usepackage{gensymb}',
                            r'\usepackage{upgreek}'],
    'xtick.major.size': 4,  # default 3.5
    'xtick.minor.size': 3,  # default 2
    'xtick.major.top': False,
    'xtick.minor.top': False,
    'xtick.minor.visible': True,
    'ytick.major.size': 4,
    'ytick.minor.size': 3,
    'ytick.major.right': False,
    'ytick.minor.right': False,
    'ytick.minor.visible': True,
})
MODE_TEXT = ['(fit)', '(manual)']


def Identity(obj: Any) -> Any:
    '''Identity
    @param obj
    @returns obj
    '''
    return obj


def power_set(index: Union[List, Tuple],
              min_length: int = 0,
              no_empty_set: bool = False):
    '''power_set'''
    assert isinstance(index, (list, tuple)), TypeError('index: list|tuple')
    for i in range(min_length, len(index) + 1):
        for x in combinations(index, i):
            if no_empty_set and not x:
                continue
            yield x


def get_doc(obj, fail: str = '', ts=4):
    if not hasattr(obj, '__doc__'):
        return fail
    s = str(obj.__doc__).rstrip().replace('\t', ' ' * ts).split('\n')
    if len(s) < 2:
        return str(obj.__doc__)
    indent = min(len(x) - len(x.lstrip()) for x in s[1:])
    return '\n'.join([s[0]] + [x[indent:] for x in s[1:]])


def intersection_interval(data: Dict[str, List],
                          selection_index: List[str],
                          upper_limit: bool = True,
                          lower_limit: bool = True) -> Tuple[Any, Any]:
    assert len(selection_index) > 1
    assert all(x in data for x in selection_index)
    sets = tuple(set(data[k]) for k in selection_index)
    intersection = sets[0].intersection(*sets[1:])
    if not intersection:
        return (0, 0)
    lb = min(intersection) if lower_limit else -np.inf
    ub = max(intersection) if upper_limit else np.inf
    return (lb, ub)


def interval_select(data: Dict[str, List],
                    selection_index: List[str],
                    interval: Tuple[Any, Any],
                    strip_NaN_values: bool = False) -> Dict[str, List]:
    '''
    Replaces all values x∉I=(a,b) with NaN
    '''
    out = {}
    lb, ub = interval
    for k in selection_index:
        tmp = []
        for v in data[k]:
            if lb < v < ub:
                tmp.append(v)
            else:
                tmp.append(np.NaN)
        out[k] = strip_NaN(tmp) if strip_NaN_values else tmp
    return out


def minmax(obj, stretch: Union[Tuple[float, float], float] = None) -> Tuple[float, float]:
    if stretch is None:
        stretch = 1.1
    if isinstance(obj, MeasurementSeries):
        mean, err, _ = obj.decompose_to_tuple()
        min_val = np.min(np.subtract(mean, err))
        max_val = np.max(np.add(mean, err))
    elif isinstance(obj, (list, tuple, np.ndarray)):
        min_val, max_val = np.min(obj), np.max(obj)
    else:
        raise TypeError('MinMax: unknown type %r' % type(obj))
    center = (min_val + max_val) / 2
    diff = (max_val - min_val) / 2
    lsc, rsc = stretch if isinstance(stretch, tuple) else (stretch, stretch)
    return center - lsc * diff, center + rsc * diff


def strip_NaN(data: List) -> List:
    non_NaN_found = False
    out = []
    buf = []
    for i, x in enumerate(data):
        if np.isnan(x):
            if non_NaN_found:
                buf.append(x)
        else:
            if not non_NaN_found:
                non_NaN_found = True
            if buf:
                out += buf
                buf = []
            out.append(x)
    return out


def multi_intersecting_values(data: Dict[str, List],
                              no_zero_length_data: bool = True,
                              strip_NaN_values: bool = True,
                              upper_limit: bool = True,
                              lower_limit: bool = True) -> Dict[Tuple, Dict[str, List]]:
    out = {}
    for p in power_set(list(data.keys()), no_empty_set=True, min_length=2):
        ii = intersection_interval(data, p, upper_limit, lower_limit)
        d = interval_select(data, p, ii, strip_NaN_values)
        if any(np.NaN in x for x in d.values()):
            # That is bad! Are those curves monotone?
            continue
        if no_zero_length_data and not any(x for x in d.values()):
            continue
        out[p] = d
    return out


def _list_or_val_to_array(obj: Union[List, np.ndarray, Any],
                          length: int,
                          dtype: Type = None) -> np.ndarray:
    if isinstance(obj, np.ndarray):
        if len(obj) == length:
            return obj
        raise ValueError('%r has %d elements but %d are required' % (obj, len(obj), length))
    elif isinstance(obj, list):
        if len(obj) == length:
            return np.array(obj, dtype=dtype)
        raise ValueError('%r has %d elements but %d are required' % (obj, len(obj), length))
    else:
        return np.full(length, obj)


#@_check
def read_csv(path: str,
             end: str = '\n',
             datasep: str = ',',
             decimalsep: str = '.',
             ignore_empty: bool = True,
             collapse_single_column: bool = True,
             headers: Union[bool, List[Hashable]] = False,
             types: Union[Type, Func, List[Type], List[Func]] = Identity,
             strip: Union[bool, List[bool]] = False,
             *args, **kwargs) -> Dict[Hashable, List]:
    '''read_csv(path, ...) -> dict'''
    # TODO: docstring
    assert end in '\r\n', ValueError(r'end must be explicitly specified: \r|\n|\r\n')
    lists = [x for x in [headers, types, strip] if isinstance(x, list)]
    assert all(len(x) == len(lists[0]) for x in lists[1:]), \
        ValueError('Different lengths specified, check headers, types, strip args!')
    n = len(end)
    d = {}
    firstline = True
    if isinstance(headers, list):
        h = headers # headers dict keys
        for x in h:
            d[x] = []
        firstline = False
    if os.path.isfile(path):
        with open(path, newline=end) as f:
            while True:
                l = f.readline()
                if not l:  # l is empty, no newline
                    break
                if ignore_empty and not l[:-n]:  # without the newline
                    continue
                p = l[:-n].split(datasep)
                if isinstance(strip, bool) and strip:
                    p = [x.strip() for x in p]
                elif isinstance(strip, list):
                    for i in range(len(p)):
                        if strip[i]:
                            p[i] = p[i].strip()
                if firstline:
                    firstline = False
                    if headers:
                        h = p  # headers dict keys
                    else:
                        h = list(range(len(p)))
                        for i in h:
                            d[i] = []
                    for x in h:
                        d[x] = []
                elif isinstance(types, list):
                    for i in range(len(p)):
                        d[h[i]].append(types[i](p[i]))
                else:
                    for i in range(len(p)):
                        d[h[i]].append(types(p[i]))
    if len(d) == 1:
        return list(d.values())[0]
    return d


def read_csvs(d: Dict[str, str],
              collapse_single_column: bool = True,
              *read_csv_args, **read_csv_kwargs) -> Dict[str, Union[Dict, List]]:
    '''read_csvs
    @param d
    @type dict
    @struct name: path

    @returns dict
    @struct name: dict|list
    '''
    n = {}
    for k, v in d.items():
        tmp = read_csv(v, *read_csv_args, **read_csv_kwargs)
        if collapse_single_column and len(tmp) == 1:
            n[k] = tmp.values()[0]
        else:
            n[k] = tmp
    return n


def multi_single_column_to_tex_table(data: Dict[str, List[Union[int, float]]],
                                     path: str,
                                     index_name: str = 'index',
                                     fmt: str = '{}',
                                     NA: str = 'NA',
                                     decimalsep: str = ',') -> Void:
    with open(path + '.tex' * (not path.endswith('.tex')), 'w') as f:
        names = [index_name] + list(data.keys())
        f.write('\\tablehead{\\hline %s \\\\\\hline\\hline}\n' % \
                ' & '.join(k.replace('&', '\\&') for k in names))
        f.write('\\begin{supertabular}{|%s|}\n' % '|'.join('c' for k in names))
        for i, tup in enumerate(zip_longest(*tuple(data.values()))):
            strings = [str(i)]
            for v in tup:
                if v is None:
                    strings.append(NA)
                else:
                    strings.append(fmt.format(v).replace('&', r'\&').replace('\\', r'\\'))
                    if decimalsep != '.':
                        strings[-1] = strings[-1].replace('.', decimalsep)
            f.write('%s \\tabularnewline\\hline\n' % ' & '.join(strings))
        f.write('\\end{supertabular}\n')


def dict_to_tex_table(data: Dict[str, List[Union[int, float, Any]]],
                      path: str,
                      vborders: bool = None,
                      columns: list = None,
                      rename_columns: dict = None,
                      index_name: str = None,
                      fmt: Union[str, List] = '{}',
                      NA: str = 'NA',
                      decimalsep: str = ',') -> Void:
    if columns is None:
        cols = list(data.keys())
    else:
        cols = [x for x in columns if x in data]
    print(index_name, index_name is not None)
    coldata =  [[1 + x for x in range(max(len(data[c]) for c in cols))]] * (index_name is not None) + [data[x] for x in cols]
    print(len(coldata), len(cols))
    if isinstance(rename_columns, dict):
        cols = [rename_columns.get(k, k) for k in cols]
    names = [index_name] * (index_name is not None) + cols
    if vborders is None:
        vborders = len(names) > 2
    vc = '|' * vborders
    with open(path + '.tex' * (not path.endswith('.tex')), 'w') as f:
        f.write('\\tablehead{\\hline %s \\\\\\hline\\hline}\n' % \
                ' & '.join(str(k).replace('&', '\\&') for k in names))
        f.write('\\begin{supertabular}{%s%s%s}\n' % (vc, vc.join('c' for k in names), vc))
        for i, ntup in enumerate(zip_longest(*tuple(coldata))):
            s = []
            for c, v in enumerate(ntup):
                cf = fmt if isinstance(fmt, str) else fmt[c]
                if v is None:
                    s.append(NA)
                else:
                    s.append(cf.format(v).replace('&', r'\&'))
                    if decimalsep != '.':
                        s[-1] = s[-1].replace('.', decimalsep)
            f.write('%s \\tabularnewline\\hline\n' % ' & '.join(s))
        f.write('\\end{supertabular}\n')


def rel_to_abs_err(data, relerr):
    assert len(data) == len(relerr)
    return np.array([data[i] * relerr[i] for i in range(len(data))], dtype=float)


def manual_R_squared(y_input, y_model):
    residuals = y_input - y_model
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y_input - np.mean(y_input)) ** 2)
    return 1 - (ss_res / ss_tot)


def R_squared_from_fit(x: Union[List, np.ndarray],
                       y: Union[List, np.ndarray],
                       f: Func,
                       pars: Tuple) -> float:
    '''R_squared_from_fit'''
    residuals = y - f(x, *pars)
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return 1 - (ss_res / ss_tot)


def fit_uncertainty(parameters: Tuple,
                    covariance_matrix: np.ndarray) -> float:
    '''fit_uncertainty'''
    err = []
    for i in range(len(parameters)):
        try:
            err.append(covariance_matrix[i][i] ** 0.5)
        except Exception as e:
            err.append(0)
    return np.array(err, dtype=float) #complex?


def fit_func(x: Union[List, np.ndarray],
             y: Union[List, np.ndarray],
             function: Func,
             parameters: Union[Tuple, List],
             yerr: Union[None, float, List, np.ndarray] = None,
             abs_err: bool = True,
             bounds: Tuple[Union[List, np.ndarray], Union[List, np.ndarray]] = (-np.inf, np.inf),
             *a, **kw) -> (Tuple, float, float, np.ndarray):
    '''fit_func(x, y, function, parameters, yerr, *a, **kw)
    returns
      + fitted parameters
      + R^2
      + fit uncertainty
      + covariance matrix
    '''
    # TODO rel y err
    if isinstance(yerr, (list, tuple)):
        if len(yerr) != len(x):
            raise ValueError('Shape mismatch (yerr): %r, %r' % (len(x), len(yerr)))
    elif isinstance(yerr, (int, float)):
        if isinstance(x, (list, tuple, range)):
            yerr = np.full(len(x), yerr)
        else:
            yerr = np.full(x.shape, yerr)
    try:
        fPar, fCovar = curve_fit(function,
                                 x, y,
                                 p0=parameters,
                                 sigma=yerr, absolute_sigma=abs_err,
                                 bounds=bounds,
                                 *a, **kw)
        # print('(fit_func) found parameters', fPar)
        # print('(fit_func) covariance')
        # pprint(fCovar)
        r_squared = R_squared_from_fit(x, y, function, tuple(fPar))
    except Exception as e:
        from traceback import print_exc as _pexc
        _pexc()
        return e, -np.inf, np.inf, None
    return tuple(fPar), R_squared_from_fit(x, y, function,  fPar), fit_uncertainty(fPar, fCovar), fCovar


def fit_and_plot(data: dict,
                 title: str,
                 xlabel: str,
                 ylabel: str,
                 xlim: Tuple[float, float] = None,
                 ylim: Tuple[float, float] = None,
                 colors: list = None,
                 figsize: Tuple[float, float] = (9, 5.5),
                 axes: plt.Axes = None,
                 center_axes: Dict[str, Union[bool, str]] = None,
                 general_plotting_options: dict = None,
                 *args,
                 **kwargs) -> (plt.Figure, plt.Axes):
    '''fit_and_plot

    Warning: This function manipulates the data dict!

    Parameters
    ----------
    data : dict
           Format: {name: {}}, name will be the legend entry for the data points

    Format of the data dict:
    {
        name: {
            'y': list|array
            opt 'x': list|array
            opt 'xerr': None|numeric|list
            opt 'yerr': None|numeric|list
            opt 'is_abs_err': bool=True
            opt 'label': str, if is does not exist: name
            opt 'fit': dict|list[dict] with the following structure:
            {
                 'f': function
                 'par': tuple of initial guesses for the parameters, will be overridden!
                 'bounds': (scalar|list, scalar|list)
                 'label': format string for the plot legend (avail: R_squared, uncertainty, p[0...], M)
                 'contx': continous x data
                 opt 'args': args to pass to physikpraktikum.fit_func
                 opt 'kwargs': args to pass to physikpraktikum.fit_funct
            }
            opt 'scatterplot_options': dict of options, see the documentation of matplotlib's plot method.
            opt 'errorbar_options': dict of options, see the documentation of matplotlib's errorbar
            opt 'fitplot_options': dict of options, see matplotlib's plot.

    Returns
    -------
    '''
    if not isinstance(colors, list):
        colors = COLORS
    gpo = general_plotting_options if isinstance(general_plotting_options, dict) else {}
    if not isinstance(center_axes, dict):
        center_axes = {  # TODO: make this global
            'left': 'auto',
            'bottom': 'auto',
            'top': 'auto',
            'right': 'auto',
        }

    if not isinstance(axes, plt.Axes):
        fig, ax = plt.subplots(figsize=figsize)
    else:
        ax = axes
        fig = plt.figure(max(plt.get_fignums()))  # find the newest figure
    i = -1
    if len(data) < 2:
        fstyles = [{'color': a1, 'linestyle': a2} for a1, a2 in combinations(COLORS + LINESTYLES, 2)
                   if a1 in a1 in COLORS and a2 in LINESTYLES]
        fstyles = sorted(fstyles, key=lambda d: d['linestyle']) # python now preserves the order of dicts, let's prefer colors

    for k, v in data.items():
        if 'color' in v:
            color = v['color']
        elif 'color' not in v.get('scatterplot_options', {}):
            i += 1
            color = colors[i % len(colors)]
        else:
            color=None
        y = v['y']
        x = v.get('x')
        if x is None:
            x = np.array(range(len(y)))

        if xlim is None:
            xlim = minmax(x, v.get('xlim_stretch'))
        if ylim is None:
            ylim = minmax(y, v.get('ylim_stretch'))

        xerr, yerr = None, None
        if isinstance(x, MeasurementSeries):
            x, xerr, *_ = x.decompose_to_tuple()
        if v.get('xerr') is not None and len(v['xerr']) == len(x):
            xerr = v['xerr']
        if isinstance(y, MeasurementSeries):
            y, yerr, *_ = y.decompose_to_tuple()
        if v.get('yerr') is not None and len(v['yerr']) == len(y):
            yerr = v['yerr']

        # scatterplot
        scpo = v.get('scatterplot_options', {})
        ascpo = {**SCATTERPLOT_OPTIONS, 'color': color, **scpo}
        ax.plot(x, y, label=v.get('label', k), **ascpo)

        # errorbars
        ebo = v.get('errorbar_options', {})
        aebo = {**ERRORBAR_OPTIONS, 'color': color, **ebo}
        if xerr is not None:
            xerr = _list_or_val_to_array(xerr, len(y), float)
        if yerr is not None:
            yerr = _list_or_val_to_array(yerr, len(y), float)
        ax.errorbar(x, y, xerr=xerr, yerr=yerr, **aebo)

        # fit
        if xlim:
            if 0 not in xlim and np.log10(xlim[1] / xlim[0]) > 1:
                contx = np.logspace(*map(np.log10, xlim), 100)
            else:
                contx = np.linspace(*xlim, 1000)
        if v.get('fit'):
            fits = v.get('fit')
            if not isinstance(fits, list):
                fits = list(fits)
            for fit_index, fit in enumerate(fits):
                if not fit.get('manual', False):
                    bounds = fit['bounds']
                    if len(bounds) != 2:
                        bounds = np.transpose(bounds)
                    par, R_sq, uncertainty, covar_mat = fit_func(x, y,
                                                                 fit['f'], fit['par'],
                                                                 v.get('yerr'),
                                                                 v.get('is_abs_err'),
                                                                 bounds,
                                                                 *fit.get('args', ()),
                                                                 **fit.get('kwargs', {}))
                else:
                    par = tuple(tmpval.n if isinstance(tmpval, UFloat) else tmpval
                                for tmpval in fit['par'])
                    if all(isinstance(tmpval, UFloat) for tmpval in fit['par']):
                        par_ufloat = fit['par']
                    elif isinstance(fit.get('par_ufloat'), (list, tuple)):
                        par_ufloat = fit['par_ufloat']
                    else:
                        par_ufloat = [ufloat(x, 0) if not isinstance(x, UFloat) else x
                                      for x in par]
                    R_sq = R_squared_from_fit(x, y, fit['f'], par)
                    uncertainty = float('nan')
                    covar_mat = np.full((len(par), len(par)), np.inf)
                if isinstance(par, Exception):
                    fit['error'] = par
                    continue
                if len(data) > 1:
                    fstyle = {'color': color, 'linestyle': LINESTYLES[fit_index % len(LINESTYLES)]}
                else:
                    fstyle = fstyles[(fit_index) % len(fstyles)]
                if not isinstance(uncertainty, float):
                    par_ufloat = tuple(map(lambda t: ufloat(*t), zip(par, uncertainty)))
                elif not all(isinstance(obj, UFloat) for obj in par_ufloat):
                    par_ufloat = [ufloat(par, np.inf) for x in par]
                fit['par'] = par
                fit['covariance_matrix'] = covar_mat
                fit['R^2'] = R_sq
                fit['par_uncertainty'] = uncertainty
                fit['par_ufloat'] = par_ufloat
                fpo = v.get('fitplot_options', {})
                spfo = fit.get('fitplot_options', {})
                afpo = {**FITPLOT_OPTIONS, **gpo, **fpo, **spfo}
                flfmt = fit.get('label', get_doc(fit['f'], ''))
                info_dict = {
                    'p': par, 'par': par,
                    'P': par_ufloat, 'par_ufloat': par_ufloat,
                    'R': R_sq, 'R_squared': R_sq,
                    'u': uncertainty, 'uncertainty': uncertainty,
                    'M': MODE_TEXT[int(fit.get('manual', False)) % 2],
                    'pi': np.pi,
                    'np': np
                }
                if sys.version_info[0] < 3 or sys.version_info[1] < 6:
                    ## Old: format stuff
                    fle = flfmt.format(**info_dict)
                else:
                    ## New: fstrings
                    fle = eval('f%r' % flfmt, info_dict, info_dict)
                ax.plot(
                    fit.get('contx', contx), fit['f'](fit.get('contx', contx), *par),
                    label=fle, **{**fstyle, **afpo})
    ax.legend()
    ax.set_xlabel(xlabel, zorder=50)  # zorder ignored
    ax.set_ylabel(ylabel, zorder=50)  # zorder ignored
    ax.set_title(title, zorder=50)
    for k, v in center_axes.items():
        spine_attrs = set(v.lower().split(' ')) if isinstance(v, str) else {v}
        if 'auto' in spine_attrs:
            i = sorted([ylim, xlim][k in ['left', 'right']]) # pyplot accepts intervals like (0, -1)
            if isinstance(i, list) and i[0] <= 0 <= i[1]:
                spine_attrs |= {'zero', 'thick', 'solid'}
            else:
                spine_attrs |= {'dash'}
        ax.spines[k].set(
            linewidth=1.5 if 'thick' in spine_attrs else 1,
            linestyle='-' if 'dash' not in spine_attrs else (0, (3, 5)),
            alpha=0.5 if 'dash' in spine_attrs else (0 if 'off' in spine_attrs else 1),
        )
        if 'zero' in spine_attrs:
            ax.spines[k].set(position='zero')
    if xlim:
        ax.set_xlim(*xlim)
    if ylim:
        ax.set_ylim(*ylim)
    return fig, ax
