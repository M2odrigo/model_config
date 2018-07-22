import numpy as np
import os
import pandas
from keras.models import Sequential
from keras.layers import Dense

def simple_dnn (filename):
    #60 * 208 hidden 0
    #30 * 208 hidden 1
    epochs = [20,40,60,80,100]
    batch = 50
    dataset = np.loadtxt(filename, delimiter=",")
    accuracy = []
    if (filename == 'data/hidden_0_activations.csv'):
        cant_input = 60
    else:
        cant_input = 30
    X = dataset[:,0:cant_input]
    Y = dataset[:,cant_input]
    for e in epochs:
        model = Sequential()
        model.add(Dense(1, input_dim=int(cant_input),activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        
        print("Configuracion de la red: ", model.summary())
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        print("entrenando para ", e, " epochs")
        model.fit(X, Y, epochs=e, batch_size=int(batch))
        scores = model.evaluate(X, Y)
        score = scores[1]*100
        print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
        accuracy.append(score)
    print(accuracy)
    df = pandas.DataFrame(accuracy)
    fn='data/acc_hidden' + str(cant_input) +'.csv'
    if os.path.isfile(fn):
        dff = pandas.read_csv(fn, header=None)
        ts = dff.append(df, ignore_index=True)
        ts.to_csv(fn, sep=',', header=None, index=None)
    else:
        dfs = df
        #print(dfs.values)
        #input('holis')
        dfs.to_csv(fn, sep=',', header=None, index=None)

#simple_dnn('data/hidden_1_activations.csv')
