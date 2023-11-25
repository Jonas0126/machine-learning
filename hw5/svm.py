import numpy as np

from libsvm.svmutil import *
from libsvm.svm import *

numData = 5000
numBit = 784


def readData(path):
    d = []
    with open(path, 'r') as file:
        for l in file.readlines():
            d.append(l.strip().strip('\n').split(','))
    d = np.array(d, dtype='f')
    
    return d

def loadData():
    
    x_train = readData('data/X_train.csv')
    y_train = readData('data/Y_train.csv')
    x_test = readData('data/X_test.csv')
    y_test = readData('data/Y_test.csv')
    return x_train, y_train, x_test, y_test


x_train, y_train, x_test, y_test = loadData()
y_train = y_train.reshape(-1).tolist()
y_test = y_test.reshape(-1).tolist()
x_train = x_train.tolist()
x_test = x_test.tolist()
print(y_train)

linearModel = svm_train(y_train, x_train, '-t 0')
polynomialModel = svm_train(y_train, x_train,'-t 1')
rbfModel = svm_train(y_train, x_train,'-t 2')

p_labs, p_acc, p_vals = svm_predict(y_test, x_test, linearModel)
print(f'acc of linear kernel : {p_acc}')
p_labs, p_acc, p_vals = svm_predict(y_test, x_test, polynomialModel)
print(f'acc of polynomial kernel : {p_acc}')
p_labs, p_acc, p_vals = svm_predict(y_test, x_test, rbfModel)
print(f'acc of rbf kernel : {p_acc}')
