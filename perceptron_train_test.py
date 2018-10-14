import numpy as np
import csv
import os
import pandas
from sklearn.preprocessing import LabelEncoder

def perceptron_train(filename, filename_test, cant_input, epochs, eta, mode):
    '''
    '''
    dataframe = pandas.read_csv(filename, header=None)
    dataset = dataframe.values
    X = dataset[:,0:int(cant_input)].astype(float)
    Y = dataset[:,int(cant_input)]
    encoderY = LabelEncoder()
    encoderY.fit(Y)
    encoded_Y = encoderY.transform(Y)
    Y = encoded_Y
    ########
    dataframes = pandas.read_csv(filename_test, header=None)
    datasets = dataframes.values
    X_test = datasets[:,0:int(cant_input)].astype(float)
    Y_test = datasets[:,int(cant_input)]
    encoder = LabelEncoder()
    encoder.fit(Y_test)
    encoded_Y_test = encoder.transform(Y_test)
    Y_test = encoded_Y_test
    ########
    Y[Y == 0] = -1
    acc = []
    for h in range(1):
        mala_clasif = []
        for v in range(1000):
            ##SI LOS WEIGHTS SON INICIALIZADOS EN ZEROS, SE SELECCIONA EL TERCER ITEM DE LA LISTA, LOS PRIMEROS DARAN 100 DEBIDO A LA RESTA 
            ##SI LOS WEIGHTS SON INICIALIZADOS DIFERENTES A ZEROS, SE SELECCIONA EL PRIMER ITEM DE LA LISTA
            #w = np.zeros(len(X[0]))
            w = np.random.uniform(0,0.5,len(X[0]))
            selected_item = 0
            eta = eta
            n = epochs
            errors = []
            for t in range(n):
                total_error = 0
                cont_error = 0
                cont_total_input=0
                for i, x in enumerate(X):
                    cont_total_input=cont_total_input+1
                    if (np.dot(X[i], w)*Y[i]) <= 0.0:
                        total_error += (np.dot(X[i], w)*Y[i])
                        w = w + eta*X[i]*Y[i]
            if(mode=='train-train'):   
                for i, x in enumerate(X):
                    if (np.dot(X[i], w)*Y[i]) <= 0.0:
                        cont_error = cont_error +1
                mala_clasif.append(cont_error)
            if(mode=='train-test'):
                for i, x in enumerate(X_test):
                    if (np.dot(X_test[i], w)*Y_test[i]) <= 0.0:
                        cont_error = cont_error +1
                mala_clasif.append(cont_error)
 
        minimo = (sum(mala_clasif) / len(mala_clasif))
        mn = 100 - (minimo*100/cont_total_input)
        acc.append(mode)
        acc.append(mn)
    
    df = pandas.DataFrame(acc)
    fn='data/perc_accuracy_crudo.csv'
    if os.path.isfile(fn):
        dff = pandas.read_csv(fn, header=None)
        ts = dff.append(df, ignore_index=True)
        ts.to_csv(fn, sep=',', header=None, index=None)
    else:
        dfs = df
        dfs.to_csv(fn, sep=',', header=None, index=None)

perceptron_train('dataset/sonar-train.csv', 'dataset/sonar-test.csv', 60, 100, 1, 'train-test')
perceptron_train('dataset/sonar-train.csv', 'dataset/sonar-train.csv', 60, 100, 1, 'train-train')

