import sys
import configparser
import numpy as np
import csv
import os
import pandas
from construct_dnn import construct_dnn
from resume_function import resume_function 
from sklearn.preprocessing import LabelEncoder
from keras import regularizers

#Leemos las configuraciones, almacenamos en variables locales para construir la red
config = configparser.ConfigParser()
config.read('config.ini')

cant_input = config['ints']['cant_input']
cant_neuronas = (config['ints']['cant_neuronas']).split(',')
cant_capas = np.arange(0, (int(config['ints']['cant_capas'])))
batch_size = config['ints']['batch_size']
cant_epochs = (config['ints']['cant_epochs']).split(',')
cant_ejecucion = config['ints']['cant_ejecucion']

activations = (config['strings']['activation']).split(',')
optimizer = config['strings']['optimizer']
loss = config['strings']['loss']

dropout=(config['dropout']['dropout_value']).split(',')

mode = config['archives']['mode']

X_test = Y_test = None

if(len(cant_capas)!=len(dropout)):
    print('dropout y cant de capas no son iguales')
    sys.exit('error, parando ejecucion')

if(mode=='train'):
    dataframe = pandas.read_csv(config['archives']['dataset'], header=None)
    dataset = dataframe.values
    X = dataset[:,0:int(cant_input)].astype(float)
    Y = dataset[:,int(cant_input)]
else:
    if(mode=='train-test'):
        dataframe = pandas.read_csv(config['archives']['dataset_train'], header=None)
        dataset = dataframe.values
        X = dataset[:,0:int(cant_input)].astype(float)
        Y = dataset[:,int(cant_input)]
        dataframe = pandas.read_csv(config['archives']['dataset_test'], header=None)
        dataset = dataframe.values
        X_test = dataset[:,0:int(cant_input)].astype(float)
        Y_test = dataset[:,int(cant_input)]
        encoder = LabelEncoder()
        encoder.fit(Y)
        encoded_Y = encoder.transform(Y_test)
        Y_test = encoded_Y

encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
Y = encoded_Y

def delete_data(cant_ejecucion):
    if os.path.isfile('data/dnn_accuracy.csv'):
        os.remove('data/dnn_accuracy.csv')
    if os.path.isfile('data/resume/resume.csv'):
        os.remove('data/resume/resume.csv')
    for i in (np.arange(int(cant_ejecucion))):
        name = 'data/perc_hidden' + str(i) +'.csv'
        if os.path.isfile(name):
            os.remove(name)

#eliminamos archivos de la ejecucion anterior
delete_data(cant_ejecucion)
####Ejecutamos N cantidad de epochs
for epochs in cant_epochs:
    fields=[('epochs'),str(epochs)]
    with open('data/resume/resume.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow(fields)
    ####Ejecutamos la red N cantidad de veces
    for i in (np.arange(int(cant_ejecucion))):
        construct_dnn(X, encoded_Y, int(cant_input), cant_capas, cant_neuronas, int(epochs), int(batch_size), activations, optimizer, loss, dropout, X_test, Y_test)
    resume_function(config['ints']['cant_capas'])

    
