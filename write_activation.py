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

def write_activation(cant_capas, cant_neuronas, cant_input, epoch, X, Y, activations):
    for capa in cant_capas:
        if (capa==0):
            model = get_model('my_model', epoch)
            #print(str(model.layers[capa].get_config()))
            test = model.layers[capa].get_config()
            #print(test)
            #print(test['name'])
            #print('dropout' not in test['name'])
            if('dropout' not in test['name']):
                activaciones = get_activations(cant_neuronas[capa], cant_input, model.layers[capa].get_weights(), X, activations[capa])
                save_activation (epoch, cant_neuronas[capa], activaciones, Y, capa, cant_capas[-1])
        else:
            model = get_model('my_model', epoch)
            #print(str(model.layers[capa].get_config()))
            test = model.layers[capa].get_config()
            #print(test)
            #print(test['name'])
            #print('dropout' not in test['name'])
            if('dropout' not in test['name']):
                #print('cant_neuronas[capa-1] ' + str(cant_neuronas[capa-1]) + '  capa ' + str(capa))
                activ_hidden = get_activations(cant_neuronas[capa], cant_neuronas[capa-1], model.layers[capa].get_weights(), activaciones, activations[capa])
                save_activation (epoch, cant_neuronas[capa], activ_hidden, Y, capa, cant_capas[-1])
                activaciones = activ_hidden


def get_model (modelName, epoch):
    print('usando el modelo y cargando weights de: ' + 'data/check/weights.'+str(epoch)+'.hdf5')
    model = load_model('data/check/'+modelName+'.h5')
    model.load_weights('data/check/weights.'+str(epoch)+'.hdf5')
    #print("Configuracion de la red: ", model.summary())
    return model
