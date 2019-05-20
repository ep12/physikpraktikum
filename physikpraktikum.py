'''PhysikPraktikum tools'''

import os
from itertools import combinations, permutations, zip_longest as zip_longest
from typing import Union, Any, Hashable, Tuple, List, List, Dict, Type, Set
from typing import NoReturn as Void, Callable as Func, Optional as Opt

# from typecheck import typecheck as _check #broken

import numpy as np
from scipy.optimize import curve_fit
import matplotlib
from matplotlib import pyplot as plt


SCATTERPLOT_OPTIONS = {
    'marker': 'x',
    'markersize': 1.5,
    'linewidth': 0,
}
FITPLOT_OPTIONS = {
    'marker': None,
    'linewidth': 1
}
ERRORBAR_OPTIONS = {
    'fmt': 'none',
    'elinewidth': 1,
    'alpha': 0.5,
}
COLORS = ['black', 'blue', 'red', 'green', 'magenta', 'cyan', 'orange']
SCISTYLE = matplotlib.RcParams(**{ # see matplotlib.pyplot.style.library['classic']
    'axes.grid': True,
    'axes.grid.axis': 'both',
    'axes.grid.which': 'both',  # major, minor, both
    'axes.spines.top': False,  # Boxplot lines on the top and right edge
    'axes.spines.right': False,
    'figure.titlesize': 18,
    'figure.subplot.left': 0.05,
    'figure.subplot.right': 0.95,
    'figure.subplot.top': 0.95,
    'figure.subplot.bottom': 0.05,
    'figure.subplot.hspace': 0.15,
    'figure.subplot.wspace': 0.15,
    'font.size': 12.0,
    'grid.linestyle': '-',
    'legend.fontsize': 10,
    'legend.loc': 'best',
    'legend.labelspacing': 0.75,
    #'savefig.format': 'svg',
    'savefig.pad_inches': 0.025,
    'text.usetex': True,
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
            return np.ndarray(obj, dtype=dtype)
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
                if isinstance(types, list):
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


def rel_to_abs_err(data, relerr):
    assert len(data) == len(relerr)
    return np.array([data[i] * relerr[i] for i in range(len(data))], dtype=float)


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
                 xlim: Tuple[float, float],
                 ylim: Tuple[float, float],
                 colors: list=None,
                 figsize: Tuple[float, float] = (9, 6.5),
                 center_axes: Dict[str, Union[bool, str]] = None,
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
            opt 'fit': {
                 'f': function
                 'par': tuple of initial guesses for the parameters, will be overridden!
                 'bounds': (scalar|list, scalar|list)
                 'label': format string for the plot legend (avail: R_squared, uncertainty, p[0...])
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
    if not isinstance(center_axes, dict):
        center_axes = {'left': 'auto', 'bottom': 'auto'}

    fig, ax = plt.subplots(figsize=figsize)
    i = -1
    for k, v in data.items():
        i += 1
        color = colors[i]
        y = v['y']
        x = v.get('x')
        if x is None:
            x = np.array(range(len(y)))

        # scatterplot
        scpo = v.get('scatterplot_options', {})
        ascpo = {**SCATTERPLOT_OPTIONS, **scpo}
        ax.plot(x, y, label=v.get('label', k), color=color, **ascpo)

        # errorbars
        if any([v.get('xerr') is not None, v.get('yerr') is not None]):
            ebo = v.get('errorbar_options', {})
            aebo = {**ERRORBAR_OPTIONS, **ebo}
            xerr, yerr = None, None
            if v.get('xerr') is not None:
                xerr = _list_or_val_to_array(v['xerr'], len(y), float)
            if v.get('yerr') is not None:
                yerr = _list_or_val_to_array(v['yerr'], len(y), float)
            ax.errorbar(x, y, xerr=xerr, yerr=yerr, color=color, **aebo)

        # fit
        if v.get('fit'):
            fit = v.get('fit')
            par, R_sq, uncertainty, covar_mat = fit_func(x, y,
                                                         fit['f'], fit['par'],
                                                         v.get('yerr'),
                                                         v.get('is_abs_err'),
                                                         fit['bounds'],
                                                         *fit.get('args', ()),
                                                         **fit.get('kwargs', {}))
            if isinstance(par, Exception):
                fit['error'] = par
                continue
            fit['par'] = par
            fit['covariance_matrix'] = covar_mat
            fit['R^2'] = R_sq
            fit['par_uncertainty'] = uncertainty
            fpo = v.get('fitplot_options', {})
            afpo = {**FITPLOT_OPTIONS, **fpo}
            fle = fit['label'].format(p=par, par=par,
                                      R=R_sq, R_squared=R_sq,
                                      u=uncertainty, uncertainty=uncertainty)
            ax.plot(
                fit['contx'], fit['f'](fit['contx'], *par),
                label=fle, color=color, **afpo)
    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    for k, v in center_axes.items():
        if v == True:
            ax.spines[k].set_position('zero')
        if v == 'auto':
            i = [ylim, xlim][k in ['left', 'right']]
            if i[0] <= 0 <= i[1]:
                # The 0 line is in the displayed interval, make it thick!
                ax.spines[k].set(position='zero', linewidth=1.5, linestyle='-')
            else:
                # The 0 line is not in the displayed interval, dash it!
                ax.spines[k].set(alpha=0.5, linewidth=1, linestyle=(0,(7, 8)))
    if xlim:
        plt.xlim(*xlim)
    if ylim:
        plt.ylim(*ylim)
    return fig, ax
