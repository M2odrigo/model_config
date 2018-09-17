import pandas as pd
import numpy as np
import os

def append_function(cant_ejec):
    for i in (np.arange(int(cant_ejec))):
        df_dnn = pd.read_csv('dnn_accuracy_delta'+str(i)+'.csv', header=None)
        df_ph0 = pd.read_csv('perc_hidden0_delta'+str(i)+'.csv', header=None)
        df_ph1 = pd.read_csv('perc_hidden1_delta'+str(i)+'.csv', header=None)
        #print(df_dnn)
        #input('okdnn')
        appe = df_dnn.append(df_ph0, ignore_index=True)
        #print(appe)
        #input('ok0')
        appe = appe.append(df_ph1, ignore_index=True)
        #print(appe)
        #input('ok1')
        #print(appe)
        append_column(appe, i)

def append_column(dataframe, i):
    if os.path.isfile('deltas.csv'):
        dff = pd.read_csv('deltas.csv', header=None)
        df_join = dff.join(dataframe, lsuffix=str(i-1)+'_exec', rsuffix=str(i)+'_exec')
        #print(df_join)
        #print('##')
        #print(df_join.astype('float'))
        df_join.to_csv('deltas.csv', sep=',', header=None, index=None)
    else:
        dataframe.to_csv('deltas.csv', sep=',', header=None, index=None)
    

append_function(50)
