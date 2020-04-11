from keras.models import load_model
import pandas
import csv
import numpy as np
from numpy import array
import sys, getopt
from urllib.parse import urlparse
import socket

import os



def load_final_model():
    return load_model('../../models/model_final.h5')

def convert_to_binary(ip_adress):
    padded_binary = ""
    for part in ip_adress.split('.'):
        binary = bin(int(part)).replace("0b", '')
        padded_binary = padded_binary + binary.rjust(8, '0')
    return padded_binary

def get_IP(hostaddress):
    return  socket.gethostbyname(hostaddress)


def run_example(argv):
    minValue = 1.0
    maxValue = 30.0
    diff = float(maxValue) - float(minValue)
    url = sys.argv[1:]
    url = str(url[0])
    parsed = urlparse(url)
    hostadd = parsed.netloc
    binaryip = convert_to_binary(get_IP(hostadd))
    split = lambda x: [int(i) for i in x]
    iparr = split(binaryip)
    X = array(iparr)
    basic_model = load_final_model()
    x_input = X
    x_input = x_input.reshape((1, 32, 1))
    result = basic_model.predict([x_input])
    predicted = result[0][0]
    denormalized = (float(diff) * float(predicted)) + float(minValue)
    print('throughput:', denormalized)


if __name__ == "__main__":
    os.environ['KMP_DUPLICATE_LIB_OK']='True'
    run_example(sys.argv)