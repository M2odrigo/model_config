import pandas as pd
import csv
import numpy as np
def resume_function(cant_ejecucion, cant_layer):
    #calculamos el AVG de los acc de la DNN
    dnn_avg = get_average('data/dnn_accuracy_resume', cant_ejecucion)
    fields=['dnn',str(dnn_avg)]
    with open('data/resume/resume.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow(fields)
    #calculamos el AVG de los acc del perceptron, cant_layer -1 para no calcular para el output layer
    for i in (np.arange((int(cant_layer)-1))):
        name = 'data/perc_hidden'+str(i)+'_resume' 
        perc_avg = get_average(name, cant_ejecucion)
        fields=[('ph'+str(i)),str(perc_avg)]
        with open('data/resume/resume.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerow(fields)
    
        
def get_average(filename, cant_ejecucion):
    df = pd.read_csv(filename+'.csv', header=None)
    dfs = pd.DataFrame(index=None, columns=None)
    columna = df.values
    #print(columna)
    #print('cant_ejecucion '+ cant_ejecucion)
    #print(type(columna))
    avg = np.divide(columna,int(cant_ejecucion))
    #print(avg)
    #input('')
    return avg
