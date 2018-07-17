from sklearn.preprocessing import LabelEncoder
import sklearn.linear_model as lm
import pandas
import configparser

config = configparser.ConfigParser()
config.read('config.ini')
cant_input = config['ints']['cant_input']
dataframe = pandas.read_csv('output_2.csv', header=None)
dataset = dataframe.values
X = dataset[:,0:30].astype(float)
Y = dataset[:,30]
encoder = LabelEncoder()
encoder.fit(Y)
y = encoder.transform(Y)

for c in range(10):
    model = lm.LogisticRegression(C=1e20, max_iter=c)
    model.fit(X, y)
    preds = model.predict(X)
    print((preds == y).mean())
