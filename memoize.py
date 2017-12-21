from collections import MutableMapping
import sqlite3, pickle, os, functools, GPflow, inspect


class PersistentDict(MutableMapping):
    '''
    From
    https://stackoverflow.com/questions/9320463/persistent-memoization-in-python
    '''
    def __init__(self, dbpath, iterable=None, **kwargs):
        self.dbpath = os.path.join('cache',dbpath) + '.cache'
        with self.get_connection() as connection:
            cursor = connection.cursor()
            cursor.execute(
                'create table if not exists memo '
                '(key blob primary key not null, value blob not null)'
            )
        if iterable is not None:
            self.update(iterable)
        self.update(kwargs)

    def encode(self, obj):
        return pickle.dumps(obj)

    def decode(self, blob):
        return pickle.loads(blob)

    def get_connection(self):
        return sqlite3.connect(self.dbpath, timeout=300)

    def  __getitem__(self, key):
        key = self.encode(key)
        with self.get_connection() as connection:
            cursor = connection.cursor()
            cursor.execute(
                'select value from memo where key=?',
                (key,)
            )
            value = cursor.fetchone()
        if value is None:
            raise KeyError(key)
        return self.decode(value[0])

    def __setitem__(self, key, value):
        key = self.encode(key)
        value = self.encode(value)
        with self.get_connection() as connection:
            cursor = connection.cursor()
            cursor.execute(
                'insert or replace into memo values (?, ?)',
                (key, value)
            )

    def __delitem__(self, key):
        key = self.encode(key)
        with self.get_connection() as connection:
            cursor = connection.cursor()
            cursor.execute(
                'select count(*) from memo where key=?',
                (key,)
            )
            if cursor.fetchone()[0] == 0:
                raise KeyError(key)
            cursor.execute(
                'delete from memo where key=?',
                (key,)
            )

    def __iter__(self):
        with self.get_connection() as connection:
            cursor = connection.cursor()
            cursor.execute(
                'select key from memo'
            )
            records = cursor.fetchall()
        for r in records:
            yield self.decode(r[0])

    def __len__(self):
        with self.get_connection() as connection:
            cursor = connection.cursor()
            cursor.execute(
                'select count(*) from memo'
            )
            return cursor.fetchone()[0]

import json
import numpy as np
from datetime import date, datetime, timedelta

def json_serial(obj):
    '''
    JSON serializer for objects not serializable by default json code
    '''

    if isinstance(obj, (datetime, date)):
        serial = obj.isoformat()
        return serial
    elif isinstance(obj, timedelta):
        serial = str(obj)
        return serial
    elif isinstance(obj, GPflow.gpr.GPR):
        d = {'__class__':obj.__class__.__name__, 
            '__module__':obj.__module__,
            'kern': str(obj.__dict__['kern']),
            'mean_function': str(obj.__dict__['mean_function']),
            'X': str(obj.__dict__['X']),
            'Y': str(obj.__dict__['Y'])}
        return d
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.int64):
        return int(obj)
    else:
        d = { '__class__':obj.__class__.__name__, 
            '__module__':obj.__module__,
            '__repr__': object.__repr__(obj)
            }
        return d
    raise TypeError ("Type %s not serializable" % type(obj))

def persistent_memoize(file_name):
    '''
    Based on 
    https://stackoverflow.com/questions/16463582/memoize-to-disk-python-persistent-memoization
    '''
    cache = PersistentDict(file_name)

    def decorator(func):
        def new_func(*args, **kwargs):
            key = (json.dumps(args, sort_keys=True, default=json_serial), 
                json.dumps(kwargs, sort_keys=True, default=json_serial))
            if key not in cache:
                cache[key] = func(*args, **kwargs)
            return cache[key]
        return new_func

    return decorator

def memoize():
    '''
    Based on 
    https://stackoverflow.com/questions/16463582/memoize-to-disk-python-persistent-memoization
    '''
    cache = {}

    def decorator(func):
        def new_func(*args, **kwargs):
            key = (json.dumps(args, sort_keys=True, default=json_serial), 
                json.dumps(kwargs, sort_keys=True, default=json_serial))
            if key not in cache:
                cache[key] = func(*args, **kwargs)
            return cache[key]
        return new_func

    return decorator
