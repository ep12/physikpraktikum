from os.path import isfile, isdir, curdir, realpath, join, commonpath
from glob import glob, iglob, has_magic
from typing import Tuple, Dict, Union

from physikpraktikum.measured.fileio import DBFile
from physikpraktikum.measured.measurement_series import MeasurementSeries


DEFAULT_SETTINGS = {
    # prefer runtime-only representations over modifying the original file
    'fileio_virtualtables_prefer': True, # after doing calculations
}
_OPTIONVALUE_MAPS = {
    'True': True, 'true': True, 'TRUE': True,
    'False': False, 'false': False, 'FALSE': False,
}


def _parse_opts(options: str):
    opts = {}
    parts = options.split(';')
    for p in parts:
        if '=' in p:
            x, y = p.split('=', 1)
            k, v = x.strip(), y.strip()
            opts[k] = _OPTIONVALUE_MAPS.get(v, v)
        else:
            opts[p.strip()] = True
    return opts


def _search_one_file(path, opts):
    if not has_magic(path):
        if isfile(path):
            return path
        return
    res = [x for x in glob(path) if isfile(x)]
    if len(res) == 1:
        return res[0]
    return


def _check_softperm_accessible(filepath, root):
    if not commonpath([root, realpath(join(root, filepath))]).startswith(root):
        raise PermissionError('Trying to access a file outside of the root directory!')

def _parse_opath(path: str, root: str) -> Tuple[Dict, Tuple]:
    '''parse a string to a file path + column
    \x00opts\x00 e.g. virtual-file (creation)
    '''
    raw = path
    filepath, column = '', ''
    if path.startswith('\x00') and path.rfind('\x00') > 0:
        i = path.rfind('\x00')
        opts = path[1: i]
        path = path[i + 1:]
        opts = {**DEFAULT_SETTINGS, **_parse_opts(opts)}
    else:
        opts = DEFAULT_SETTINGS
    if path.startswith('/'):
        path = path[1:]
    filepath, column = _search_one_file(path, opts), ''
    if not filepath:
        if '/' in path:
            filepath, column = path.rsplit('/', 1)
            filepath = _search_one_file(filepath, opts)
    if not filepath:
        raise FileNotFoundError('Could not find a file in %r (root directory: %r)' % (path, root))
    _check_softperm_accessible(filepath, root)
    return opts, (realpath(join(root, filepath)), column)


class Database:
    def __init__(self,
                 root_directory: str = None,
                 **kwargs):
        '''Database

        Additional keyword arguments are stored as self.settings, see
        physikpraktikum.measured.database.DEFAULT_SETTINGS for supported options
        '''
        if root_directory is None:
            root_directory = realpath(curdir)
        assert isdir(root_directory), ValueError('root_directory must be a directory')
        self.root = root_directory
        self.settings = kwargs
        self._fcache = {}

    def __str__(self):
        return '<Database root=%r />' % self.root

    def __len__(self):
        '''Returns the number of cached MeasurementSeries'''
        return len(self._fcache)

    def __getitem__(self, index: str) -> Union[DBFile, MeasurementSeries]:
        opts, (fpath, col) = _parse_opath(index, self.root)
        self._ensure_cached(fpath, **opts)
        return self._fcache[fpath][col]

    def __setitem__(self, index: str, value):
        opts, (fpath, col) = _parse_opath(index, self.root)
        self._ensure_cached(fpath, **opts)
        #print(fpath)
        #return self._fcache[fpath][col] = value

    def to_cache(self, path, *args, **kwargs):
        assert isfile(path), FileNotFoundError(path)
        self._fcache[path] = DBFile(path, *args, **kwargs)

    def _ensure_cached(self, path, *args, **kwargs):
        if path not in self._fcache:
            self.to_cache(path, *args, **kwargs)


if __name__ == '__main__':
    #x = '/../../brokenaxes.py'
    #x = '../../brokenaxes.py'
    x = '\x00cache=True\x00../../physikpraktikum/measured/__init__.py'
    db = Database()
    print(db[x])

