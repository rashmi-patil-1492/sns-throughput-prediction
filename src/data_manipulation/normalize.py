import pandas as pd
import csv
import numpy


def load_file(file_path):
    data_frame = pd.read_csv(file_path, header=None, delimiter=',')
    return data_frame.values


devset_data = load_file('../../data/devset.csv')
testset_data = load_file('../../data/testset.csv')
devset_outfile = '../../data/devset_normalized.csv'
testset_outfile = '../../data/testset_normalized.csv'


# based on outliers.py following and min and max values of throughput in the whole dataset.
minValue = 1.0
maxValue = 30.0

with open(devset_outfile, 'w') as outfilename:
    writer = csv.writer(outfilename)
    for line in devset_data:
        normalized = str((float(line[2]) - minValue) / (maxValue - minValue))
        writer.writerow(numpy.append(line, normalized))
print('dev set Done !!!')


with open(testset_outfile, 'w') as outfilename:
    writer = csv.writer(outfilename)
    for line in testset_data:
        normalized = str((float(line[2]) - minValue) / (maxValue - minValue))
        writer.writerow(numpy.append(line, normalized))
print('test set Done !!!')