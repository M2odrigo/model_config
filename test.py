import pandas as pd
import os
import numpy as np
def sum_columns (filename):
    df = pd.read_csv(filename+'.csv', header=None)
    dfs = pd.DataFrame(index=None, columns=None)
    columna = df.values
    print(columna)
    print(type(columna))
    print(np.divide(columna,3))
    input('some')

sum_columns('data/perc_hidden1_resume')
