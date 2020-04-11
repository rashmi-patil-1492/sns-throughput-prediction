import pandas
from numpy import array
from math import sqrt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, Bidirectional
from keras.layers import Dropout
import csv
import numpy as np
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score


dataframe = pandas.read_csv("../../data/devset_normalized.csv", delim_whitespace=False, delimiter=',', header=None)
dataset = dataframe.values
split = lambda x: [int(i) for i in x]
X = [split(bit_vector) for bit_vector in [row[1] for row in dataset if len(row[1]) == 32]]
X_Original =[row for row in dataset]
Y = [float(tput) for tput in [row[6] for row in dataset if len(row[1]) == 32]]

def mae_metric(actual, predicted):
    sum_error = 0.0
    for i in range(len(actual)):
        sum_error += abs(predicted[i] - actual[i])
    return sum_error / float(len(actual))


def mse_metric(actual, predicted):
    sum_error = 0.0
    for i in range(len(actual)):
        prediction_error = predicted[i] - actual[i]
        sum_error += (prediction_error ** 2)
    mean_error = sum_error / float(len(actual))
    return (mean_error)

def rmse_metric(actual, predicted):
    sum_error = 0.0
    for i in range(len(actual)):
        prediction_error = predicted[i] - actual[i]
        sum_error += (prediction_error ** 2)
    mean_error = sum_error / float(len(actual))
    return sqrt(mean_error)

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


X, Y = array(X), array(Y)
X = X.reshape((X.shape[0], X.shape[1], 1))

kfold = KFold(n_splits=10, random_state=None, shuffle=False)

minValue = 1.0
maxValue = 30.0

diff = float(maxValue) - float(minValue)

cnt = 0
for train, test in kfold.split(X, Y):

    cnt = cnt + 1
    basic_model = cnn_lstm_model()
    basic_model.fit(X[train], Y[train], epochs=25, verbose=2, validation_data=[X[test], Y[test]], batch_size=16)

    original_values_from_test = [X_Original[test_index] for test_index in test]

    y_actual_normalized = []
    y_actual_original = []
    y_hat_normalized = []
    y_hat_denormalized = []
    for i in range(len(X[test])):
        x_input = X[test][i]
        x_input = x_input.reshape((1, 32,1))
        result = basic_model.predict([x_input])
        predicted = result[0][0]
        denormalized = (float(diff) * float(predicted)) + float(minValue)
        y_hat_normalized.append(predicted)
        y_hat_denormalized.append(denormalized)
        y_actual_normalized.append(float(original_values_from_test[i][6]))
        y_actual_original.append(float(original_values_from_test[i][2]))


    print(['result_printed', cnt,
           mae_metric(y_actual_normalized, y_hat_normalized),
           mse_metric(y_actual_normalized, y_hat_normalized),
           rmse_metric(y_actual_normalized, y_hat_normalized),
           r2_score(y_actual_normalized, y_hat_normalized)])

    outfile = '../../outputs/disagreements/output_disagreement_'+str(cnt)+'.csv'
    with open(outfile, 'w') as outfilename:
        writer = csv.writer(outfilename)
        for i in range(len(X[test])):
            normalized_diff = abs(float(original_values_from_test[i][6]) - float(y_hat_normalized[i]))
            denormalized_diff = abs(float(original_values_from_test[i][2]) - float(y_hat_denormalized[i]))
            writer.writerow(np.append(original_values_from_test[i], [y_hat_normalized[i], normalized_diff,  y_hat_denormalized[i], denormalized_diff]))
        print('Done !!!')

