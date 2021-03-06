import numpy as np
import csv
import os
import pandas

def perceptron_train(filename, filename_test, layer, cant_input, epochs, eta, dnn_epoch):
    '''
    '''
    dataset = np.loadtxt(filename, delimiter=",")
    # split into input (X) and output (Y) variables
    X = dataset[:,0:cant_input]
    #add the bias term -1
    X = np.insert(X, cant_input,-1, axis=1)
    Y = dataset[:,cant_input]
    Y[Y == 0] = -1
    acc = []
    maximos = []
    for h in range(2):
        mala_clasif = []
        for v in range(500):
            ##SI LOS WEIGHTS SON INICIALIZADOS EN ZEROS, SE SELECCIONA EL TERCER ITEM DE LA LISTA, LOS PRIMEROS DARAN 100 DEBIDO A LA RESTA 
            ##SI LOS WEIGHTS SON INICIALIZADOS DIFERENTES A ZEROS, SE SELECCIONA EL PRIMER ITEM DE LA LISTA
            #w = np.zeros(len(X[0]))
            w = np.random.uniform(0,0.5,len(X[0]))
            selected_item = 0
            #print(w)
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
                
            for i, x in enumerate(X):
                if (np.dot(X[i], w)*Y[i]) <= 0.0:
                    cont_error = cont_error +1
            mala_clasif.append(cont_error)
        #print(mala_clasif)
        #print('#$%%$##')
        #input('okokokok')
        
        #en caso de que se necesite acceder al momento donde el perceptron tuvo menos errores, se ordena la lista de menor a mayor.
        #elegimos un valor luego de las primeras tres, para evitar enganos por la inicializacion en zeros
        ###Respecto al peso (w), se le pasa el ultimo actualizado, si se necesitara pasar donde el error es minimo, utilizar el array de pesos (weights) y acceder mediante el indice
        #print(len(mala_clasif))
        #print('###sorting..')
        #print(mala_clasif)
        minimo = (sum(mala_clasif) / len(mala_clasif))
        mn = 100 - (minimo*100/cont_total_input)
        #a modo de aplicar correctamente la metrica, se selecciona el ultimo valor de la lista de errores, y se pasan esos weights para la prediccion
        #mn = mala_clasif[-1]
        #print(minimo)        
        #print(mn)
        #print(cont_total_input)
        #input('ok?')
        #error_minimo = mn/cont_total_input
        #accuracy = 100 -(error_minimo*100)
        #print('accuracy:: ' + str(accuracy))
        acc.append(mn)
        acc.append(dnn_epoch)
        mala_clasif.sort()
        mx = mala_clasif[selected_item]
        mx = 100 - (mx*100/cont_total_input)
        maximos.append(mx)
        maximos.append(dnn_epoch)
        
    
    #print(acc)
    #input('tabien?')
    #fields=[layer,str(cont_total_input),accuracy]
    #campo = [accuracy]
    #name = 'data/perc_hidden' + str(layer) +'.csv'
    df = pandas.DataFrame(acc)
    #print(df)
    #input('okokoko')
    #df.to_csv('data/perc_hidden' + str(layer) +'.csv', header=None, index=None)
    fn='data/perc_hidden' + str(layer) +'.csv'
    if os.path.isfile(fn):
        dff = pandas.read_csv(fn, header=None)
        #print(dff)
        #columna2 = dff.values
        #dfs = df-dff  ##restar las columnas
        #dfs = dff.join(df, lsuffix='_exec', rsuffix='other_exec')
        ts = dff.append(df, ignore_index=True)
        #print(ts)
        #print(columna2)
        #input('test')
        ts.to_csv(fn, sep=',', header=None, index=None)
    else:
        dfs = df
        #print(dfs.values)
        #input('holis')
        dfs.to_csv(fn, sep=',', header=None, index=None)
    
    dfmx = pandas.DataFrame(maximos)
    fnmx='data/perc_max_hidden' + str(layer) +'.csv'
    if os.path.isfile(fnmx):
        dffmx = pandas.read_csv(fnmx, header=None)
        tsmx = dffmx.append(dfmx, ignore_index=True)
        tsmx.to_csv(fnmx, sep=',', header=None, index=None)
    else:
        dfsmx = dfmx
        dfsmx.to_csv(fnmx, sep=',', header=None, index=None)
    #with open(name, 'a') as f:
    #    writer = csv.writer(f)
    #    writer.writerow(campo)
    #with open('data/train_hidden_perceptron_error.csv', 'a') as f:
    #    writer = csv.writer(f)
    #    writer.writerow(fields)
    ##print(" error minimo "+ str(mn) + ' acc. '  +str(accuracy) + ' total entradas: ' + str(cont_total_input))
    #perceptron_prediction(filename_test, layer, cant_input, w)
    #input('wait')


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
    #print('####predcciones perceptron###')
    with open('data/prediction_hidden_perceptron_error.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow(fields)

#w = np.zeros(60)
#print(w);
#input('ok?')
#w1 = np.random.uniform(0,0.5,60)
#print(w1)
#input('ok??')
