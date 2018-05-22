import numpy as np
import csv

def perceptron_train(filename, filename_test, layer, cant_input, epochs, eta):
    '''
    '''
    dataset = np.loadtxt(filename, delimiter=",")
    # split into input (X) and output (Y) variables
    X = dataset[:,0:cant_input]
    #add the bias term -1
    X = np.insert(X, cant_input,-1, axis=1)
    Y = dataset[:,cant_input]
    Y[Y == 0] = -1
    w = np.zeros(len(X[0]))
    eta = eta
    n = epochs
    errors = []
    mala_clasif = []

    for t in range(n):
        total_error = 0
        cont_error = 0
        cont_total_input=0
        for i, x in enumerate(X):
            cont_total_input=cont_total_input+1
            if (np.dot(X[i], w)*Y[i]) <= 0.0:
                total_error += (np.dot(X[i], w)*Y[i])
                w = w + eta*X[i]*Y[i]
                cont_error = cont_error +1
        errors.append(total_error*-1)
        mala_clasif.append(cont_error)
    
    mala_clasif.sort()
    print(mala_clasif)
    #elegimos un valor luego de las primeras tres, para evitar enganos por la inicializacion en zeros
    mn = mala_clasif[3]
    error_minimo = mn/cont_total_input
    accuracy = 100 -(error_minimo*100)
    fields=[layer,str(cont_total_input),accuracy]
    with open('data/train_hidden_perceptron_error.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow(fields)
    #print(" error minimo "+ str(mn) + ' acc. '  +str(accuracy) + ' total entradas: ' + str(cont_total_input))
    perceptron_prediction(filename_test, layer, cant_input, w)


def perceptron_prediction(filename, layer, cant_input, w):
    dataset = np.loadtxt(filename, delimiter=",")
    # split into input (X) and output (Y) variables
    X = dataset[:,0:int(cant_input)]
    #add the bias term -1
    X = np.insert(X, int(cant_input),-1, axis=1)
    Y = dataset[:,int(cant_input)]
    Y[Y == 0] = -1
    eta = 1
    total_error = 0
    cont_error = 0
    cont_total_input=0
    for i, x in enumerate(X):
        cont_total_input=cont_total_input+1
        if (np.dot(X[i], w)*Y[i]) <= 0.0:
            cont_error = cont_error +1
    accuracy = 100-((cont_error*100)/cont_total_input)
    fields=[layer,str(cont_total_input),accuracy]
    print('####predcciones perceptron###')
    with open('data/prediction_hidden_perceptron_error.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow(fields)