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

#leemos el archivo de configuracion
config = configparser.ConfigParser()
config.read('config.ini')

def construct_dnn (X, Y, cant_input, cant_capas, cant_neuronas, cant_epochs, batch_size, activations, optimizer, loss, dropout, intervalo, delta, X_test=None, Y_test=None, wsave = None):
    #eliminamos archivos temporales
    delete_data()
    #recorrer las capas para ir configurando la red
    #print(dropout)
    regularizador=get_configurations()
    kernel = convert_regularization(regularizador[0], 'kernel_l_value')
    bias = convert_regularization(regularizador[1], 'bias_l_value')
    activity = convert_regularization(regularizador[2], 'activity_l_value')
    activaciones = []
    model = Sequential()
    if(regularizador!=''):
        for capa in cant_capas:
            if capa == 0:
                model.add(Dense(int(cant_neuronas[capa]), input_dim=int(cant_input), activation=activations[capa], kernel_regularizer=kernel, activity_regularizer=activity, bias_regularizer=bias))
                #print(dropout[capa])
                #input('drop')
                if(float(dropout[capa]) > 0):       
                    model.add(Dropout(float(dropout[capa])))         
            else:
                model.add(Dense(int(cant_neuronas[capa]), activation=activations[capa],kernel_regularizer=kernel, activity_regularizer=activity, bias_regularizer=bias))
                #print(dropout[capa])
                #input('drop')
                if(float(dropout[capa]) > 0):       
                    model.add(Dropout(float(dropout[capa]))) 
    else:
        for capa in cant_capas:
            if capa == 0:
                model.add(Dense(int(cant_neuronas[capa]), input_dim=int(cant_input), activation=activations[capa]))
                if(float(dropout[capa]) > 0):       
                    model.add(Dropout(float(dropout[capa]))) 
            else:
                model.add(Dense(int(cant_neuronas[capa]), activation=activations[capa]))
                if(float(dropout[capa]) > 0):       
                    model.add(Dropout(float(dropout[capa]))) 
    model.save('data/check/my_model.h5')
    if not (wsave):
        print("vamos a guardar los pesos de la primera ejecucion")
        wsave=model.get_weights()
    else:
        print("vamos a cargar los pesos para esta ejecucion")
        model.set_weights(wsave)
    #print(model.get_weights())
    #input('verificar pesos ')
    print("Configuracion de la red: ", model.summary())
    #agregamos un callback para entrar dentro del metodo fit() y extraer datos
    weight_save_callback = ModelCheckpoint('data/check/weights.{epoch:02d}.hdf5', monitor='val_loss', verbose=0, save_best_only=False, mode='auto', period=intervalo)
    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
    model.fit(X, Y, epochs=cant_epochs, batch_size=batch_size, verbose=2,callbacks=[weight_save_callback])
    #print('###PREDICTION###')
    if(X_test.any()):
        # evaluate the model
        print('vamos a entrenar sobre un dataset y testear sobre otro')
        scores = model.evaluate(X_test, Y_test)
    else:
        # evaluate the model
        scores = model.evaluate(X, Y)
    acc = ("%.2f%%" % (scores[1]*100))
    acc_temp = scores[1]*100
    print('Accuracy ' + str(acc_temp))
    ##print(model.get_config())
    #fields=[str(cant_epochs),str(acc), model.get_config()]
    #header = ['epochs', 'acc', 'metodo']
    #if os.path.isfile('dnn_acc.csv'):
    #    os.remove('dnn_acc.csv')
    #with open('dnn_acc.csv', 'a') as f:
    #    writer = csv.writer(f)
    #    writer.writerow(header)
    #    writer.writerow(fields)
    #vamos almacenando el accuracy de la red en cada iteracion (por cada ejecucion)
    campo = [str(acc_temp)]
    print('###########ACCURRACY' + str(acc_temp))
    with open('data/dnn_accuracy.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow(campo)
    return wsave

def get_activations (cant_nodos, cant_input, weights, activations, activation):
    model = Sequential()
    model.add(Dense(int(cant_nodos), input_dim=int(cant_input), weights=weights, activation=activation))
    activations = model.predict_proba(activations)
    activations_array = np.asarray(activations)
    return activations_array

def get_configurations ():
    regularizador = [None, None, None]
    kernel_regularizer = config['regularizers']['kernel_regularizer']
    bias_regularizer = config['regularizers']['bias_regularizer']
    activity_regularizer = config['regularizers']['activity_regularizer']

    if (str(kernel_regularizer)=='true'):
        kernel_value = config['regularizers']['kernel_value']
        regularizador[0]= kernel_value

    if (bias_regularizer=='true'):
        bias_value = config['regularizers']['bias_value']
        regularizador[1]= bias_value

    if (activity_regularizer=='true'):
        activity_value = config['regularizers']['activity_value']
        regularizador[2]= activity_value

    #print(regularizador)
    return regularizador

def convert_regularization(regularization, value):
    #print('recibimos: ' + str(regularization))
    v = config['regularizers'][value]
    r=None
    #print('value to l : ' + v)
    if (regularization=='l2'):
        r = regularizers.l2(v)
    elif (regularization=='l1'):
        r = regularizers.l1(v)
    elif (regularization=='l1_l2'):
        r = regularizers.l1_l2(v)
    return r

def delete_data():
    if os.path.isfile('data/train_hidden_perceptron_error.csv'):
        os.remove('data/train_hidden_perceptron_error.csv')
    if os.path.isfile('data/prediction_hidden_perceptron_error.csv'):
        os.remove('data/prediction_hidden_perceptron_error.csv')    

