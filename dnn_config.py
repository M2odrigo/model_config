import configparser
import numpy as np

#Leemos las configuraciones, almacenamos en variables locales para construir la red
config = configparser.ConfigParser()
config.read('config.ini')

cant_neuronas = (config['ints']['cant_neuronas']).split(',')
cant_capas = np.arange(1, (int(config['ints']['cant_capas'])+1))
batch_size = config['ints']['batch_size']

activations = (config['strings']['activation']).split(',')
optimizer = config['strings']['optimizer']
loss = config['strings']['loss']
dropout = config['strings']['dropout']


