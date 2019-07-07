'''FileIO manages IO ops (multi-file database'''
from physikpraktikum.errors import PermissionError
from pp import read_pp


class DBFile:
    def __init__(self, path: str, permissions: str = 'r',
                 *args, **kwargs):
        self.__dict__['path'] = path
        self.__dict__['permissions'] = permissions

    def __str__(self):
        return '<DBFile path=%r mode=%r>' % (self.path, self.permissions)

    def __setattr__(self, attr, val):
        if attr in 'permissions':
            raise PermissionError('You are not allowed to do this!')
        self.__dict__[attr] = val

    #def __enter__(self, *args, **kwargs):
    #    pass

    #def __exit__(self, *args, **kwargs):
    #    pass

    def __getitem__(self, index):
        if index == '' or index is None:
            return self
        # pass
        # read file!

    def __setitem__(self, index, value):
        # read file!
        pass # TODO

    def _ensure_cached(self):
        pass

    def _read_file(self):
        pass # TODO

