from keras.models import Sequential
from keras.layers import *
from keras.optimizers import *
from keras.initializers import *

import matplotlib.pyplot as plt
import numpy as np
import random

pairs = ['AUDJPY', 'AUDNZD', 'AUDUSD', 'CADJPY', 'CHFJPY', 'EURGBP', 'EURJPY', 'EURUSD', 'GBPJPY', 'GBPUSD', 'NZDUSD', 'USDCAD']
years = ['2012', '2013', '2014', '2015', '2016', '2017']
months = ['JANUARY', 'FEBRUARY', 'MARCH', 'APRIL', 'MAY', 'JUNE', 'JULY', 'AUGUST', 'SEPTEMBER', 'OCTOBER', 'NOVEMBER', 'DECEMBER']

MEMORY_CAPACITY = 1000
BATCH_SIZE = 64

GAMMA = 0.99

MAX_EPSILON = 0 # 0 for no random
LAMBDA = 0.001      # speed of decay


class Brain:
    def __init__(self, stateCnt, actionCnt, weights_h5=None):
        """Class to create and train a neural network for prediction.

        Args:
            stateCnt (int): The size of the input state vector
            actionCnt (int): The size of the output action vector
            weights_h5 (string): Filename of h5 file to load from,
                None if to not load from any file.

        """
        self.stateCnt = stateCnt
        self.actionCnt = actionCnt

        self.model = self._createModel()
        if weights_h5:
            self.model.load_weights(weights_h5)

    def _createModel(self):
        model = Sequential()

        model.add(Reshape(target_shape=(1, self.stateCnt)))
        model.add(Dense(output_dim=256, activation='relu', input_shape=(1, self.stateCnt),
            kernel_initializer='ones', bias_initializer='zeros'))
        model.add(Dense(output_dim=256, activation='relu',
            kernel_initializer='ones', bias_initializer='zeros'))
        model.add(LSTM(output_dim=256, activation='relu',
            kernel_initializer='ones', bias_initializer='zeros'))
        model.add(Dense(output_dim=self.actionCnt, activation='linear',
            kernel_initializer=RandomNormal(0, 0.001), bias_initializer='zeros'))

        opt = RMSprop(lr=0.00025)
        model.compile(loss='mse', optimizer=opt)

        return model

    def train(self, x, y, epoch=1, verbose=0, batch_size=64):
        self.model.fit(x, y, batch_size=batch_size, nb_epoch=epoch, verbose=verbose)

    def predict(self, s):
        return self.model.predict(s)

    def predictOne(self, s):
        return self.predict(s.reshape(1, self.stateCnt)).flatten()


class Memory:   # stored as ( s, a, r, s_ )
    samples = []

    def __init__(self, capacity):
        self.capacity = capacity

    def add(self, sample):
        self.samples.append(sample)

        if len(self.samples) > self.capacity:
            self.samples.pop(0)

    def sample(self, n):
        n = min(n, len(self.samples))
        return random.sample(self.samples, n)


class Agent:
    def __init__(self, stateCnt, actionCnt):
        """Class to manage an agent to train it's Brain

        Args:
            stateCnt (int): The size of the input state vector
            actionCnt (int): The size of the output action vector
            weights_h5 (string): Filename of h5 file to load from,
                None if to not load from any file.

        """
        self.stateCnt = stateCnt
        self.actionCnt = actionCnt

        self.steps = 0
        self.epsilon = MAX_EPSILON

        self.brain = Brain(self.stateCnt, self.actionCnt)
        self.memory = Memory(MEMORY_CAPACITY)

    def act(self, s):
        # Random actions
        if random.random() < self.epsilon:
            return random.randint(0, self.actionCnt - 1)

        return np.argmax(self.brain.predictOne(s))

    def observe(self, sample):  # in (s, a, r, s_) format
        self.memory.add(sample)

        # slowly decrease Epsilon based on our eperience
        self.steps += 1
        # self.epsilon = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * math.exp(-LAMBDA * self.steps)

    def replay(self):
        batch = self.memory.sample(BATCH_SIZE)
        batchLen = len(batch)

        no_state = np.zeros(self.stateCnt)

        states = np.array([o[0] for o in batch])
        states_ = np.array([(no_state if o[3] is None else o[3]) for o in batch])

        p = self.brain.predict(states)
        p_ = self.brain.predict(states_)

        x = np.zeros((batchLen, self.stateCnt))
        y = np.zeros((batchLen, self.actionCnt))

        for i in range(batchLen):
            o = batch[i]
            s = o[0];
            a = o[1];
            r = o[2];
            s_ = o[3]

            t = p[i]
            if s_ is None:
                t[a] = r
            else:
                t[a] = r + GAMMA * np.amax(p_[i])

            x[i] = s
            y[i] = t

        self.brain.train(x, y)


class Environment:
    def __init__(self, data, ds):
        """Class to manage an agent to train it's Brain

        Args:
            df (DataFrame): the training data

        """

        self.actions = {0: 'pos_short', 1: 'pos_neutral', 2: 'pos_long'}
        self.state_variables = ds.columns

        df = data.join(ds)

        df = df[~df.isnull().any(axis=1)]
        # init portfolio
        df['portfolio'] = 0.0
        df.at[df.index[0], 'portfolio'] = 1000.0
        df.pos_short = 0
        df.pos_neutral = 0
        df.pos_long = 0

        self.close_col = df.columns.get_loc('close')
        self.open_col = df.columns.get_loc('open')
        self.next_col = df.columns.get_loc('next_open')
        self.pf_col = df.columns.get_loc('portfolio')

        self.state = np.asarray(df.iloc[0][self.state_variables])

        self.df = df
        self.current_datetime = self.df.index[0]
        self.canvas = {'portfolio': [], 'next_open': [], 'index': []}

        plt.ion()
        self.fig, self.ax = plt.subplots()

    def run(self, agent):
        self.reset()
        s = self.state
        R = 0

        done = None
        while not done:
            # get the agent's action at state s
            a = agent.act(s)

            # apply the action to the environment
            s_, r, done = self.act(a)

            if done:  # terminal state
                s_ = None

            # give the agent the observation data
            agent.observe((s, a, r, s_))

            agent.replay()
            """ force the agent to relive this horrific event
            and learn from it via neural net training
            """

            s = s_  # current state is now the new state
            R += r  # we add the event's reward to our total

            self.render()  # we render the environment

        print("Total reward:", R)

    def calc_action(self, action):
        if action == 'pos_neutral':
            return 0
        if action == 'pos_short':
            return -1
        if action == 'pos_long':
            return 1

    def calc_portfolio(self, trade_size=100, spread=0.0001):
        index = self.df.index.get_loc(self.current_datetime)

        pf_col = self.df.columns.get_loc('portfolio')
        long_col = self.df.columns.get_loc('pos_long')
        short_col = self.df.columns.get_loc('pos_short')
        close_col = self.df.columns.get_loc('next_open')
        open_col = self.df.columns.get_loc('open')

        v_0 = self.df.iat[index - 1, pf_col]

        a_0 = 0
        if self.df.iat[index - 1, long_col]:
            a_0 = 1
        elif self.df.iat[index - 1, short_col]:
            a_0 = -1

        a = 0
        if self.df.iat[index, long_col]:
            a = 1
        elif self.df.iat[index, short_col]:
            a = -1

        c = self.df.iat[index, close_col]
        o = self.df.iat[index, open_col]
        d = trade_size * abs(a - a_0) * spread

        return v_0 + a * trade_size * (c - o) - d

    def _update_state(self, action):
        """
        Input: action and states
        Ouput: new states and reward
        """

        self.df.at[self.current_datetime, self.actions[action]] = 1  # Apply action
        p = self.calc_portfolio()  # Calculate the current portfolio
        self.df.at[self.current_datetime, 'portfolio'] = p  # Save it
        self.current_datetime = self._get_next_datetime(self.current_datetime)  # Increment
        return np.asarray(self.df.loc[self.current_datetime][self.state_variables])

    def _get_next_datetime(self, dt):
        _c = self.df.index.get_loc(dt)
        return self.df.index[_c + 1]

    def _get_prev_datetime(self, dt):
        _c = self.df.index.get_loc(dt)
        return self.df.index[_c - 1]

    def _get_reward(self):
        try:
            return np.log(self.df.at[self.current_datetime, 'portfolio'] /
                          self.df.at[self._get_prev_datetime(self.current_datetime), 'portfolio'])
        except IndexError as e:
            return 0

    def _is_over(self):
        try:
            self._get_next_datetime(self.current_datetime)
            return False
        except IndexError as e:
            return True
        #return self.df.at[self.current_datetime, 'next_open'] is None

    def render(self):
        if self.current_datetime.hour or self.current_datetime.minute:
            return
        plt.clf()
        index = self.df.index.get_loc(self.current_datetime)
        pf_col = self.df.columns.get_loc('portfolio')

        cur = self.df.iloc[index - 1]
        self.canvas['index'].append(cur.name)
        self.canvas['next_open'].append(cur.next_open)
        self.canvas['portfolio'].append(cur.portfolio)

        ymin = min(self.canvas['portfolio'])
        ymax = max(self.canvas['portfolio'])
        ymax = ymin if ymin > max(self.canvas['next_open']) else max(self.canvas['next_open'])

        plt.title(str(cur.name) + ": " + str(cur.portfolio))
        plt.xlim(self.df.index[0], self.current_datetime)
        plt.plot(self.df.portfolio / 1000.0, 'b-')
        plt.plot(1 - self.df.next_open / self.df.at[self.df.index[0], 'next_open'], 'r-')
        # plt.plot(self.df.index, self.df.next_open, 'r-')
        plt.show()
        self.fig.canvas.draw()

    def act(self, action):
        self.state = self._update_state(action)
        reward = self._get_reward()
        game_over = self._is_over()
        return self.state, reward, game_over

    def reset(self):
        self.current_datetime = self.df.index[0]


if __name__ == "__main__":
    import preprocess as pp
    df = pp.get_pickle('2012')
    df = df[df.pair == "EUR/USD"]
    df['next_open'] = df.open.shift(-1)
    dff = df.copy()

    df = pp.calculate_market_variables(dff)
    df = pp.calculate_meta_variables(df)
    df['portfolio'] = 0.0

    use_columns = [col for col in df.columns if '_log' in col]
    use_columns += ['day_of_week', 'hour', 'minute', 'pos_short', 'pos_neutral', 'pos_long']

    env = Environment(df, use_columns)
    stateCnt = len(use_columns)
    actionCnt = 3

    agent = Agent(stateCnt, actionCnt)
    env.run(agent)
    agent.brain.model.save("brain.h5")
