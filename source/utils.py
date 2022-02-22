import os.path

import pandas as pd


def read_datafile(filename, has_class=True):
    filepath = os.path.join('..', 'data', filename)
    return pd.read_csv(filepath).iloc[:, :-1].to_numpy(copy=True)
