import pandas as pd
import os
def diff_columns (filename, delta):
    delta=str(delta)
    df = pd.read_csv(filename+'.csv', header=None)
    #dfs = pd.DataFrame(index=None, columns=None)
    columna = df.values
    ##esto es para otra metrica, donde se restan los ACC en cada ejecucion
    #if os.path.isfile(filename+'_delta'+delta+'.csv'):
    if os.path.isfile(filename+'_delta.csv'):
        dff = pd.read_csv(filename+'_delta.csv', header=None)
        columna2 = dff.values
        #dfs = df-dff  ##restar las columnas
        dfs = df.join(dff, lsuffix='_exec', rsuffix='other_exec')
        #print('##')
        #print(columna2)
        #input('test')
        dfs.to_csv(filename+'_delta.csv', sep=',', header=None, index=None)
    else:
        dfs = df
        #print(dfs.values)
        #input('holis')
        dfs.to_csv(filename+'_delta.csv', sep=',', header=None, index=None)

    
