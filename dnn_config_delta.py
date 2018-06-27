import sys
import configparser
import numpy as np
import csv
import os
import pandas
from construct_dnn_delta import construct_dnn
from write_activation import write_activation
from resume_function import resume_function 
from sklearn.preprocessing import LabelEncoder
from keras import regularizers
from sum_columns import sum_columns
from diff_columns import diff_columns

def dnn_config_delta(delta):
    #Leemos las configuraciones, almacenamos en variables locales para construir la red
    #Metodo para leer otra configuracion sin detener la ejecucion
    def read_config(filename):
        config = configparser.ConfigParser()
        config.read(filename)

        dropout=(config['dropout']['dropout_value']).split(',')
        return dropout

    ##CONFIGURACIONES INICIALES
    config = configparser.ConfigParser()
    config.read('config.ini')

    cant_input = config['ints']['cant_input']
    cant_neuronas = (config['ints']['cant_neuronas']).split(',')
    cant_capas = np.arange(0, (int(config['ints']['cant_capas'])))
    batch_size = config['ints']['batch_size']
    cant_epochs = int(config['ints']['cant_epochs'])
    cant_ejecucion = config['ints']['cant_ejecucion']
    intervalo = int(config['ints']['intervalo'])

    activations = (config['strings']['activation']).split(',')
    optimizer = config['strings']['optimizer']
    loss = config['strings']['loss']

    dropout=(config['dropout']['dropout_value']).split(',')

    mode = config['archives']['mode']

    X_test = Y_test = None

    if(len(cant_capas)!=len(dropout)):
        #print('dropout y cant de capas no son iguales')
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
    def delete_data_resume(cant_ejecucion):
        if os.path.isfile('data/resume/resume.csv'):
            os.remove('data/resume/resume.csv')
        if os.path.isfile('data/dnn_accuracy_resume.csv'):
            os.remove('data/dnn_accuracy_resume.csv')
        for i in (np.arange(int(cant_ejecucion)+1)):
            name = 'data/perc_hidden' + str(i) +'_resume.csv'
            if os.path.isfile(name):
                os.remove(name)

    def delete_data(cant_ejecucion):
        if os.path.isfile('data/dnn_accuracy.csv'):
            os.remove('data/dnn_accuracy.csv')
        #if os.path.isfile('data/resume/resume.csv'):
        #   os.remove('data/resume/resume.csv')
        for i in (np.arange(int(cant_ejecucion)+1)):
            name = 'data/perc_hidden' + str(i) +'.csv'
            if os.path.isfile(name):
                os.remove(name)

    #eliminamos archivos de resumen de la configuracion anterior
    delete_data_resume(cant_ejecucion)
    #eliminamos archivos de la ejecucion anterior
    delete_data(cant_ejecucion)
    wsave=None
    ####Ejecutamos la red N cantidad de veces
    for i in (np.arange(int(cant_ejecucion))):
        print('ejecucion nro ::::::::::: ' + str(i))
        if(i>0):
            filename="config"+str(i)+".ini"
            print("vamos a leer otra configuracion para " + filename)
            dropout = read_config(filename)
        wsave = construct_dnn(X, encoded_Y, int(cant_input), cant_capas, cant_neuronas, cant_epochs, int(batch_size), activations, optimizer, loss, dropout, intervalo, delta, X_test, Y_test, wsave)
        for epoch in range(intervalo,(cant_epochs+intervalo),intervalo):
            #fields=[('epochs'),str(epoch)]
            #with open('data/resume/resume.csv', 'a') as f:
                #writer = csv.writer(f)
                #writer.writerow(fields)
            write_activation(cant_capas, cant_neuronas, cant_input, epoch, X, Y, activations)
        #sum_columns('data/perc_hidden1')
        #sum_columns('data/perc_hidden0')
        #sum_columns('data/dnn_accuracy')
        diff_columns('data/perc_hidden1', delta)
        diff_columns('data/perc_hidden0', delta)
        diff_columns('data/dnn_accuracy', delta)
        #delete_data(cant_ejecucion)

    #resume_function(cant_ejecucion, config['ints']['cant_capas'])

        
