from keras.models import Sequential
from keras.layers import Dense

def construct_dnn (X, Y, cant_input, cant_capas, cant_neuronas, cant_epochs, batch_size, activations, optimizer, loss, dropout):
    #recorrer las capas para ir configurando la red
    model = Sequential()
    for capa in cant_capas:
        if capa == 0:
            model.add(Dense(int(cant_neuronas[capa]), input_dim=int(cant_input), activation=activations[capa]))
        else:
            model.add(Dense(int(cant_neuronas[capa]), activation=activations[capa]))

    print("Configuracion de la red: ", model.summary())

    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
    model.fit(X, Y, epochs=cant_epochs, batch_size=batch_size)
