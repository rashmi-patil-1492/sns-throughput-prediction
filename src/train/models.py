import pandas

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout, Bidirectional
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D


def cnn_lstm_model():
    model = Sequential(name='cnn_lstm_model')
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


if __name__ == "__main__":
    print(cnn_lstm_model().summary())