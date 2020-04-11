from keras.models import load_model
import pandas
import csv
import numpy as np
from numpy import array
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'


def mse_metric(actual, predicted):
    sum_error = 0.0
    for i in range(len(actual)):
        prediction_error = predicted[i] - actual[i]
        sum_error += (prediction_error ** 2)
    mean_error = sum_error / float(len(actual))
    return mean_error


# load model

dataframe = pandas.read_csv("../../data/testset_normalized.csv", delim_whitespace=False, delimiter=',', header=None)
dataset = dataframe.values
split = lambda x: [int(i) for i in x]
X = [split(bit_vector) for bit_vector in [row[1] for row in dataset if len(row[1]) == 32]]
X_Original =[row for row in dataset]
Y = [float(tput) for tput in [row[6] for row in dataset if len(row[1]) == 32]]

X, Y = array(X), array(Y)
X = X.reshape((X.shape[0], X.shape[1], 1))
basic_model = load_model('../../models/model_final.h5')
scores = basic_model.evaluate(X, Y, verbose=1)
print(scores)
minValue = 1.0
maxValue = 30.0
diff = float(maxValue) - float(minValue)

y_actual_normalized = []
y_actual_original = []
y_hat_normalized = []
y_hat_denormalized = []
for i in range(len(X)):
    x_input = X[i]
    x_input = x_input.reshape((1, 32, 1))
    result = basic_model.predict([x_input])
    predicted = result[0][0]
    denormalized = (float(diff) * float(predicted)) + float(minValue)
    y_hat_normalized.append(predicted)
    y_hat_denormalized.append(denormalized)
    y_actual_normalized.append(float(X_Original[i][6]))
    y_actual_original.append(float(X_Original[i][2]))

print('mse', mse_metric(y_actual_normalized, y_hat_normalized))
outfile = '../../outputs/final_output.csv'
with open(outfile, 'w') as outfilename:
    writer = csv.writer(outfilename)
    for i in range(len(X)):
        normalized_diff = abs(float(X_Original[i][6]) - float(y_hat_normalized[i]))
        denormalized_diff = abs(float(X_Original[i][2]) - float(y_hat_denormalized[i]))
        writer.writerow(np.append(X_Original[i],[y_hat_normalized[i], normalized_diff,
                                                 y_hat_denormalized[i], denormalized_diff]))
    print('Done !!!')

