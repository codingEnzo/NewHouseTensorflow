# -*- coding: utf-8 -*-
import multiprocessing as mg
import sys

import numpy as np
import pandas as pd

reload(sys)
sys.setdefaultencoding('utf8')


def apply_func(args):
    df, func, kwargs = args
    for i, colname in enumerate(df.columns):
        _mean = df[colname].mean()
        _max = df[colname].max()
        _min = df[colname].min()
        df[colname] = df.apply(lambda r: func(r[colname], _mean, _max, _min), **kwargs)
    return df


def apply_by_multiprocessing(df, func, **kwargs):
    workers = kwargs.pop('workers')
    pool = mg.Pool(processes=workers)
    result = pool.map(apply_func, [(d, func, kwargs) for d in np.array_split(df, workers)])
    pool.close()
    return pd.concat(list(result))
