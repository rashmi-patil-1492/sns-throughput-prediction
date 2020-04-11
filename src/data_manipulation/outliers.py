import pandas as pd
import csv

df = pd.read_csv("../../data/raw_data.csv", delim_whitespace=False, delimiter=',', header=None)
dataset = df.values

cnt = 0
outfile = '../../data/data.csv';
with open(outfile, 'w') as outfilename:
    writer = csv.writer(outfilename)
    for data in dataset:
        if (float(data[2]) > 30) or (float(data[2]) < 1):
            cnt = cnt + 1
            continue
        else:
            writer.writerow(data)
print('Done !!!', cnt, 'ignored out of ', len(dataset))