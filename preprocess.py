import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import zscore


def get_pickle(pairs=None, years=None, months=None, directory='data'):
    def cast_to_array(param):
        if type(param) in [str, type(None)]:
            param = [param]
        return param

    def filter_files(params, files=[]):
        _files = []
        data_files = [f for f in os.listdir('data') if '.pickle' in f]
        for param in cast_to_array(params):
            if not param:
                break
            if len(param) == 2:
                sym = '.'
            else:
                sym = '-'
            _files = [f for f in data_files if param + sym in f]
            files += _files
        return files

    files = []
    files = filter_files(pairs, files)
    files = filter_files(years, files)
    files = filter_files(months, files)

    if directory[-1] != '/':
        directory += '/'

    df = None
    for f in files:
        _df = pd.read_pickle(directory + f)
        if type(df) != pd.DataFrame:
            df = _df
        else:
            df = pd.concat([df, _df])

    df = df[df.volume > 0]
    # df['next_open'] = df.open.shift(-1)

    return df.sort_index()


def quick_plot(df):
    num_of_pairs = len(df.pair.unique())

    cols = int(num_of_pairs / (3 - 1))
    if num_of_pairs == 1:
        rows = cols = 1
    elif num_of_pairs < 3:
        rows = num_of_pairs
    else:
        rows = 3

    fig = plt.figure(figsize=(9 * rows, 6 * cols), dpi=80, facecolor='w', edgecolor='k')

    for v, pair in enumerate(df.pair.unique()):
        df_pair = df[df.pair == pair]
        ax = plt.subplot(cols, rows, v + 1)
        ax.text(0.5, 0.9, pair, horizontalalignment='center',
                verticalalignment='center', transform=ax.transAxes)
        ax.plot(df_pair.index, df_pair.close)

    plt.show()


def sin_encode(t, T):
    return np.sin(2 * np.pi * (t / T))


def encode_temporal(index, name):
    _index = index
    if name == "day_of_week":
        index = index.dayofweek
    if name == "hour":
        index = index.hour
    if name == "minute":
        index = index.minute

    T = len(index.unique())
    return pd.Series(sin_encode(index, T), index=_index, name=name)


def quick_plot_encoding(encoding, xlim=1000):
    plt.plot(encoding)
    plt.xlim(encoding.index[0], encoding.index[xlim])
    plt.show()


def log_return(series):
    return np.log(series) - np.log(series.shift(1))


def rolling_zscore(arr):
    return zscore(arr)[-1]


def rolling_zscore_alt(log_ret):
    return (log_ret - log_ret.rolling(96).mean()) / log_ret.rolling(96).std()


def gen_log_return(series, zscore_roll=96, lookback=8, clip=(-10,10)):
    log_ret = log_return(series)
    norms = log_ret.rolling(zscore_roll).apply(rolling_zscore)
    norms = norms.clip(*clip)

    log_returns = {}
    for s in range(lookback):
        log_returns['{}_log_returns_{}'.format(series.name, s)] = norms.shift(s)

    return pd.DataFrame(log_returns)


def gen_pos_encoding(df):
    return pd.DataFrame(np.zeros(shape=(df.close.size, 3)),
                      columns=['pos_short', 'pos_neutral', 'pos_long'],
                      index=df.index)


def calculate_meta_variables(dff):
    # Time Encoding
    dff = dff.join(encode_temporal(dff.index, name='day_of_week'))
    dff = dff.join(encode_temporal(dff.index, name='hour'))
    dff = dff.join(encode_temporal(dff.index, name='minute'))

    # Position Encoding
    dff = dff.join(gen_pos_encoding(dff))

    return dff


def calculate_market_variables(dff):
    # Market Feature
    dff = dff.join(gen_log_return(dff.next_open))
    dff = dff.join(gen_log_return(dff.volume))

    # Cleaning
    dff = dff.drop(columns=["high", "low"])

    return dff