import pandas as pd
import csv
import numpy as np
def resume_function(cant_layer):
    #calculamos el AVG de los acc de la DNN
    dnn_avg = get_average('data/dnn_accuracy.csv')
    fields=['dnn',str(dnn_avg)]
    with open('data/resume/resume.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow(fields)
    #calculamos el AVG de los acc del perceptron, cant_layer -1 para no calcular para el output layer
    for i in (np.arange((int(cant_layer)-1))):
        name = 'data/perc_hidden' + str(i) +'.csv'
        perc_avg = get_average(name)
        fields=[('ph'+str(i)),str(perc_avg)]
        with open('data/resume/resume.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerow(fields)
    
        
def get_average(filename):
    df = pd.read_csv(filename, header=None)
    columna = df.values
    suma = 0
    contador = 0
    for x in columna:
        contador=contador+1
        suma = suma + x
    avg = suma/contador
    #print(avg)
    return avg
