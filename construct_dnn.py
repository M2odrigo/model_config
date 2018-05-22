import numpy as np
import csv
from keras.models import Sequential
from keras.layers import Dense
from save_activations import save_activation

def construct_dnn (X, Y, cant_input, cant_capas, cant_neuronas, cant_epochs, batch_size, activations, optimizer, loss, dropout, X_test=None, Y_test=None):
    #recorrer las capas para ir configurando la red
    activaciones = []
    model = Sequential()
    for capa in cant_capas:
        if capa == 0:
            model.add(Dense(int(cant_neuronas[capa]), input_dim=int(cant_input), activation=activations[capa]))
        else:
            model.add(Dense(int(cant_neuronas[capa]), activation=activations[capa]))

    print("Configuracion de la red: ", model.summary())

    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
    model.fit(X, Y, epochs=cant_epochs, batch_size=batch_size)
    print('###PREDICTION###')
    if(X_test!=None and X_test.any()):
        # evaluate the model
        scores = model.evaluate(X_test, Y_test)
    else:
        # evaluate the model
        scores = model.evaluate(X, Y)
    acc = ("%.2f%%" % (scores[1]*100))
    print('Accuracy ' + str(acc))
    fields=[str(cant_epochs),str(acc)]
    with open('dnn_acc.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow(fields)
    for capa in cant_capas:
        if (capa==0):
            activaciones = get_activations(cant_neuronas[capa], cant_input, model.layers[capa].get_weights(), X, activations[capa])
            save_activation (cant_epochs, cant_neuronas[capa], activaciones, Y, capa, cant_capas[-1])
        else:
            activ_hidden = get_activations(cant_neuronas[capa], cant_neuronas[capa-1], model.layers[capa].get_weights(), activaciones, activations[capa])
            save_activation (cant_epochs, cant_neuronas[capa], activ_hidden, Y, capa, cant_capas[-1])
            activaciones = activ_hidden

def get_activations (cant_nodos, cant_input, weights, activations, activation):
    model = Sequential()
    model.add(Dense(int(cant_nodos), input_dim=int(cant_input), weights=weights, activation=activation))
    activations = model.predict_proba(activations)
    activations_array = np.asarray(activations)
    return activations_array
