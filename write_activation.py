from construct_dnn import get_activations
import numpy as np
import csv
import configparser
import os
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras import regularizers
from keras.layers import Dropout
from keras.models import load_model
from keras.callbacks import Callback
from keras.callbacks import ModelCheckpoint
from save_activations import save_activation

def write_activation(cant_capas, cant_neuronas, cant_input, epoch, X, Y, activations, loss, optimizer):
    conta = -1
    for capa in cant_capas:
        if (capa==0):
            #print('estamos en la capa '+ str(capa))
            model = get_model('my_model', epoch)
            get_accuracy(model, X,Y,epoch, loss, optimizer)
            #print(str(model.layers[capa].get_config()))
            test = model.layers[capa].get_config()
            #print(test)
            #print(test['name'])
            #print('dropout' not in test['name'])
            if('dropout' not in test['name']):
                activaciones = get_activations(cant_neuronas[capa], cant_input, model.layers[capa].get_weights(), X, activations[capa])
                save_activation (epoch, cant_neuronas[capa], activaciones, Y, capa, cant_capas[conta])
                conta = -1
            else:
                conta = -2
        else:
            #print('pasamos a la capa '+ str(capa))
            model = get_model('my_model', epoch)
            #print(str(model.layers[capa].get_config()))
            #model.get_layer('dense_1').get_config()
            test = model.layers[capa].get_config()
            #print(test)
            
            #print('dropout' not in test['name'])
            if('dropout' not in test['name']):
                #print(test['name'])
                #print('cant_neuronas[capa-1] ' + str(cant_neuronas[capa-1]) + '  capa ' + str(capa))
                activ_hidden = get_activations(cant_neuronas[capa], cant_neuronas[capa-1], model.layers[capa].get_weights(), activaciones, activations[capa])
                save_activation (epoch, cant_neuronas[capa], activ_hidden, Y, capa, cant_capas[-1])
                activaciones = activ_hidden

def get_model (modelName, epoch):
    #print('usando el modelo y cargando weights de: ' + 'data/check/weights.'+str(epoch)+'.hdf5')
    model = load_model('data/check/'+modelName+'.h5')
    model.load_weights('data/check/weights.'+(str(epoch).zfill(2))+'.hdf5')
    #print("Configuracion de la red: ", model.summary())
    return model

def get_accuracy (model, X, Y, epoch, loss, optimizer):
    acc=[]
    print('accuracy para epoch '+ (str(epoch).zfill(2)))
    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
    scores = model.evaluate(X, Y)
    acc_temp = scores[1]*100
    acc.append(acc_temp)
    acc.append(epoch)
    df = pandas.DataFrame(acc)
    fn='data/dnn_epoch_accuracy.csv'
    if os.path.isfile(fn):
        dff = pandas.read_csv(fn, header=None)
        ts = dff.append(df, ignore_index=True)
        ts.to_csv(fn, sep=',', header=None, index=None)
    else:
        dfs = df
        dfs.to_csv(fn, sep=',', header=None, index=None)
    #print('Accuracy ' + str(acc_temp))
