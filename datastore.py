import pandas as pd
import numpy as np
from scipy.stats import zscore
import os
import sys
import logging
import datetime
from logger import setup_custom_logger

class DataProcessor:
    def __init__(self, instance_name='', prepare_data=True):
        """Class to create, store, and fetch data
        """
        self.lg = setup_custom_logger('DataProcessor: ' + instance_name, instance_name)
        self.lg.info('DataProcessor Initializing...')

        if prepare_data:
            if os.path.isfile('data/master.pickle'):
                self.lg.info('Loading Master Data')
                self.clean_df = self.load_data('data/master.pickle')
            else:
                self.lg.error('data/master.pickle is required')
                raise FileNotFoundError('data/master.pickle is required')

            if os.path.isfile('data/state_space.pickle'):
                self.lg.info('Loading State Data')
                self.state = self.load_data('data/state_space.pickle')
            else:
                self.lg.info('Generated State Data')
                self.state = self.prepare_state_space()

        self.lg.info("DataProcessing Initialized!")

    def load_data(self, file):
        return pd.read_pickle(file)

    def get_unique_pairs(self):
        self.pairs = self.clean_df.pair.unique()
        return self.pairs

    def slice_data_by_pair(self, pair):
        return self.clean_df[self.clean_df.pair == pair]

    def gen_next_open(self, d):
        d['next_open'] = d.open.shift(-1)
        return d

    def sin_encode(self, t, T):
        return np.sin(2 * np.pi * (t / T))

    def encode_temporal(self, index, name):
        _index = index
        if name == "day_of_week":
            index = index.dayofweek
        if name == "hour":
            index = index.hour
        if name == "minute":
            index = index.minute

        T = len(index.unique())
        return pd.Series(self.sin_encode(index, T), index=_index, name=name)

    def log_return(self, series):
        return np.log(series) - np.log(series.shift(1))

    def rolling_zscore(self, arr):
        return zscore(arr)[-1]

    def gen_pos_encoding(self, dff):
        return pd.DataFrame(np.zeros(shape=(dff.index.size, 3)),
                      columns=['pos_short', 'pos_neutral', 'pos_long'],
                      index=dff.index)

    def gen_log_return(self, series, zscore_roll=96, lookback=8, clip=(-10,10)):
        log_ret = self.log_return(series)
        norms = log_ret.rolling(zscore_roll).apply(self.rolling_zscore, raw=False)
        norms = norms.clip(*clip)

        log_returns = {}
        for s in range(lookback):
            log_returns['{}_log_returns_{}'.format(series.name, s)] = norms.shift(s)

        return pd.DataFrame(log_returns)

    def calculate_meta_variables(self, dff):
        # Time Encoding
        dff = dff.join(self.encode_temporal(dff.index, name='day_of_week'))
        dff = dff.join(self.encode_temporal(dff.index, name='hour'))
        dff = dff.join(self.encode_temporal(dff.index, name='minute'))

        return dff


    def calculate_market_variables(self, dff, label=''):
        # Market Feature
        _df = self.gen_log_return(dff.next_open)
        _df = _df.join(self.gen_log_return(dff.volume))
        _df.columns = [str(col) + '_' + label for col in _df.columns]

        return _df

    def prepare_state_space(self):
        self.state = None

        self.dfps = {} # DataFrame Dictionary per Pair
        pairs = self.get_unique_pairs()

        for pair in pairs:
            self.lg.info('Preparing Data for {}'.format(pair))
            _df = self.slice_data_by_pair(pair).copy()
            _df = self.gen_next_open(_df)
            _df = self.calculate_market_variables(_df, pair.replace('/', '_'))
            self.dfps[pair] = _df
            if type(self.state) == type(None):
                self.state = _df
            else:
                self.state = self.state.join(_df)

        # Position Encoding
        self.state = self.calculate_meta_variables(self.state)
        self.state = self.state.join(self.gen_pos_encoding(self.state))
        return self.state

    def combine_state_spaces(self):
        ds = DataProcessor(False)
        df = None
        for key in ds.dfps:
            if type(df) == type(None):
                df = self.dfps[key]
            else:
                df = df.join(self.dfps[key])

        df = self.calculate_meta_variables(df)
        df = self.join(ds.gen_pos_encoding(df))
        return df
        #df.to_pickle('data/state_space.pickle')

class DataManager:
    def __init__(self, dp, instance_name=''):
        """Class to manage preprocessed data
        """

        self.lg = setup_custom_logger('DataManager: ' + instance_name, instance_name)
        self.lg.info('DataManager Initializing...')

        self.dp = dp
        self.data = dp.clean_df
        self.state = dp.state
        self.pairs = self.data.pair.unique()
        self.training_columns = self.get_training_columns()

        self.lg.info('DataManager Initialized!')

    def get_pairs(self):
        return self.pairs

    def get_training_columns(self):
        self.training_columns = self.state.columns
        return self.training_columns

    def get_slice_from_pair(self, pair):
        return self.dp.gen_next_open(self.data[self.data.pair == pair])
