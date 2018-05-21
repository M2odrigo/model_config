import os
import csv
import pandas
from split_function import split
def save_activation (e, cant_nodos, activations, Y, layer, last_layer):
    if(layer != last_layer):
        print("epoch ", e, "cant nodos: ", cant_nodos, "activation shape: ", activations.shape)
        #print(activations)
        if os.path.isfile('data/hidden_'+str(layer) +'_activations.csv'):
            os.remove('data/hidden_'+str(layer) +'_activations.csv')
        for index, activ in enumerate(Y):
            r2 = ['{:f}'.format(x) for x in activations[index]]
            fields=[r2, Y[index]]
            with open('data/hidden_'+str(layer) +'_activations.csv', 'a') as f:
                writer = csv.writer(f)
                writer.writerow(fields)

        # Read in the file
        with open('data/hidden_'+str(layer) +'_activations.csv', 'r') as file :
            filedata = file.read()
            # Replace the target string
            filedata = filedata.replace('[', '')
            filedata = filedata.replace(']', '')
            filedata = filedata.replace('"', '')
            filedata = filedata.replace('\'', '')
        # Write the file out again
        with open('data/hidden_'+str(layer) +'_activations.csv', 'w') as file:
            file.write(filedata)
        
        filename = 'data/hidden_'+str(layer) +'_activations.csv'
        shuffle(filename)
        split(open(filename, 'r'), 146);
        filename1 = 'output_1.csv'
        filename2 = 'output_2.csv'
        input('contunye_?')
        #perceptron_plot(filename1, layer, e,cant_nodos)
        #perceptron_plot(filename2, layer, e,cant_nodos)
        #input('continue?') 
    else:
        print("capa actual " + str(layer))
        print("capa final " + str(last_layer))
        print("llegamos al output layer")


def shuffle (filename):
    dataframe = pandas.read_csv(filename, header=None)
    df = dataframe.sample(frac=1)
    os.remove(filename)
    df.to_csv(filename, sep=',', header=None, index=None)
