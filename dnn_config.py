import configparser
import numpy as np
import csv
import os
import pandas
from construct_dnn import construct_dnn
from sklearn.preprocessing import LabelEncoder

#Leemos las configuraciones, almacenamos en variables locales para construir la red
config = configparser.ConfigParser()
config.read('config.ini')

cant_input = config['ints']['cant_input']
cant_neuronas = (config['ints']['cant_neuronas']).split(',')
cant_capas = np.arange(0, (int(config['ints']['cant_capas'])))
batch_size = config['ints']['batch_size']
cant_epochs = config['ints']['cant_epochs']

activations = (config['strings']['activation']).split(',')
optimizer = config['strings']['optimizer']
loss = config['strings']['loss']
dropout = config['floats']['dropout']

dataframe = pandas.read_csv(config['archives']['dataset'], header=None)
dataset = dataframe.values
X = dataset[:,0:int(cant_input)].astype(float)
Y = dataset[:,int(cant_input)]

encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)

construct_dnn(X, encoded_Y, int(cant_input), cant_capas, cant_neuronas, int(cant_epochs), int(batch_size), activations, optimizer, loss, dropout)
