import pandas
from numpy import random
from numpy import array

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout, Bidirectional
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D


# following two lines are Mac OS specfic and can be removed in other environment
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


def shuffle(df, n=1, axis=0):
    df = df.copy()
    for _ in range(n):
        df.apply(random.shuffle, axis=axis)
    return df


dataframe = pandas.read_csv("../../data/devset_normalized.csv", delim_whitespace=False, delimiter=',', header=None)
df_shuf = shuffle(dataframe)
dataset = df_shuf.values
split = lambda x: ([int(i) for i in x])
X = [split(bit_vector) for bit_vector in [row[1] for row in dataset if len(row[1]) == 32]]
Y = [float(tput) for tput in [row[6] for row in dataset if len(row[1]) == 32]]
X, Y = (array(X),  array(Y))


def baseline_model():
    model = Sequential()
    model.add(Conv1D(128, 16, activation='relu', input_shape=(32, 1)))
    model.add(MaxPooling1D(pool_size=8))
    model.add(Bidirectional(LSTM(128, kernel_initializer='glorot_uniform', activation='relu')))
    model.add(Dense(128, kernel_initializer='normal', activation='relu'))
    model.add(Dense(128, kernel_initializer='normal', activation='relu'))
    model.add(Dense(128, kernel_initializer='normal', activation='relu'))
    model.add(Dense(128, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, kernel_initializer='normal', activation='relu'))
    # Compile model
    model.compile(loss='mse', optimizer='adam')
    return model


n_features = 1
X_train = X.reshape((X.shape[0], X.shape[1], 1))

basic_model = baseline_model()
basic_model.fit(X_train, Y, epochs=25, verbose=2, batch_size=16)
basic_model.save('../../models/final_model.h5')
print("Done !!!")


