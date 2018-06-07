import pandas as pd
import os
def sum_columns (filename):
    df = pd.read_csv(filename+'.csv', header=None)
    #dfs = pd.DataFrame(index=None, columns=None)
    columna = df.values
    print(columna)

    if os.path.isfile(filename+'_resume.csv'):
        dff = pd.read_csv(filename+'_resume.csv', header=None)
        columna2 = dff.values
        dfs = df+dff
        #print('##')
        #print(columna2)
        #input('test')
        dfs.to_csv(filename+'_resume.csv', sep=',', header=None, index=None)
    else:
        dfs = df
        #print(dfs.values)
        #input('holis')
        dfs.to_csv(filename+'_resume.csv', sep=',', header=None, index=None)

    
