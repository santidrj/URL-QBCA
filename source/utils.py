import os.path

import pandas as pd


def read_datafile(filename, has_class=True):
    filepath = os.path.join("..", "data", filename)
    df = pd.read_csv(filepath)
    if has_class:
        return df.iloc[:, :-1].to_numpy(copy=True), df.iloc[:, -1].to_numpy(copy=True)

    return df.to_numpy(copy=True), None
