import csv

in_file = '../../outputs/overfitting_overfitted.txt'
lossInfo = []
with open(in_file, 'r') as inputfile:
    lines = inputfile.readlines()
    for line in lines:
        if 'val_loss' in line:
            training_loss = line[line.find('loss: ') + 6: line.find(' - val_loss')]
            val_loss = line[line.find('val_loss: ') + 10:]
            lossInfo.append([training_loss, val_loss])


outfile = '../../outputs/overfitting_overfitted.csv';
with open(outfile, 'w') as outfilename:
    writer = csv.writer(outfilename)
    for data in lossInfo:
        writer.writerow(data)
print('Done !!!')