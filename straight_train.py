import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

from keras.models import Sequential
from keras.layers import *
from keras.optimizers import *
from keras.initializers import *
from keras.utils.np_utils import to_categorical
from keras.callbacks import ModelCheckpoint

ds = pd.read_pickle('data/state_space.pickle')
ds.pos_neutral = 1.0 # assume we are always neutral
state_cols = list(ds.columns)
remove_cols = [c for c in ds.columns if '_log_returns_0' in c]

for c in remove_cols:
    state_cols.remove(c)

dm = pd.read_pickle('data/master.pickle')
spread = 0.0001

pairs = dm.pair.unique()
dps = {}
for pair in pairs:
    dps[pair] = dm[dm.pair == pair].copy()
    dps[pair]['next_open'] = dps[pair].open.shift(-1)
    dps[pair]['log_change_short'] = np.log((dps[pair].open - spread) / dps[pair].next_open)
    dps[pair]['log_change_neutral'] = 0.0
    dps[pair]['log_change_long'] = np.log(dps[pair].next_open / ( dps[pair].open + spread ))

df = dps['EUR/USD'].join(ds)[~dps['EUR/USD'].join(ds).isna().any(axis=1)]

ver_amnt = int(len(df)*0.2)

input_data = np.array(df[state_cols])[:-ver_amnt]
output_data = np.array(df[['log_change_short', 'log_change_neutral', 'log_change_long']])[:-ver_amnt]
#output_data[:,0] + (output_data[:,0] > 0.0004).astype(int)-1
#output_data[:,2] + (output_data[:,2] > 0.0004).astype(int)-1
output_data = to_categorical(np.argmax(output_data, axis=1), num_classes=3)

input_verify = np.array(df[state_cols])[-ver_amnt:]
output_verify = np.array(df[['log_change_short', 'log_change_neutral', 'log_change_long']])[-ver_amnt:]
output_verify = to_categorical(np.argmax(output_verify, axis=1), num_classes=3)

model = Sequential()

model.add(Dense(units=256, activation='elu', input_shape=input_data[0].shape))
model.add(Dense(units=256, activation='elu'))
model.add(Dense(units=256, activation='elu'))
model.add(Dense(units=len(output_data[0]), activation='softmax'))

opt = Adam(lr=0.0025)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

filepath="data/weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

history = model.fit(input_data, output_data, batch_size=1, epochs=250, callbacks=callbacks_list, verbose=1)
np.save('data/history', np.array([history.history]))
model.save('data/nn.h5')
