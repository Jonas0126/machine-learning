import numpy as np
from scipy.spatial.distance import cdist
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

def grid_search(x_train, y_train, grid, model):
    opt = {}
    if model == 'softlinear':
        f = open(model+'.txt', 'w') 
        
        best = 0 
        for i in grid['C']:
            parameter = '-t 0 -s 0 -v 5'
            parameter += ' -c ' + str(i)
            accuracy = svm_train(y_train, x_train, parameter)
            f.write(f'Parameter C : {i}, acc : {accuracy}\n')
            if accuracy > best:
                best = accuracy
                opt['C'] = str(i)
        temp = opt['C']
        f.write(f'best parameter C : {temp}, acc : {best}\n')
        return opt
    
    elif model == 'softpolynomial':
        f = open(model+'.txt', 'w')
        
        best = 0
        for i in grid['C']:
            for j in grid['gamma']:
                for k in grid['coef0']:
                    for d in grid['degree']:
                    
                        parameter = '-t 1 -s 0 -v 5 -c ' + str(i) + ' -g ' + str(j) + ' -r ' + str(k) + ' -d ' + str(d)
                        print(parameter)
                        accuracy = svm_train(y_train, x_train, parameter)
                        f.write(f'Parameter C : {i}, Parameter gamma : {j}, Parameter coef0 : {k}, Parameter degree : {d}, acc : {accuracy}\n')
                        if accuracy > best:
                            best = accuracy
                            opt['C'] = str(i)
                            opt['gamma'] = str(j)
                            opt['coef0'] = str(k)
                            opt['degree'] = str(d)
        c = opt['C']
        g = opt['gamma']
        r = opt['coef0']
        f.write(f'best parameter C : {c}, best parameter gamma : {g}, best parameter coef : {r}, acc : {best}\n')
        return opt
    
    elif model == 'softrbf':
        f = open(model+'.txt', 'w')
        
        best = 0
        for i in grid['C']:
            for j in grid['gamma']:
                parameter = '-t 2 -s 0 -v 5 -c ' + str(i) + ' -g ' + str(j)
                accuracy = svm_train(y_train, x_train, parameter)
                f.write(f'Parameter C : {i}, Parameter gamma : {j}, acc : {accuracy}\n')
                if accuracy > best:
                    best = accuracy
                    opt['C'] = str(i)
                    opt['gamma'] = str(j)
        c = opt['C']
        g = opt['gamma']
        f.write(f'Parameter C : {c}, Parameter gamma : {g}, acc : {best}\n')
        return opt
#Data process
x_train, y_train, x_test, y_test = loadData()
y_train = y_train.reshape(-1).tolist()
y_test = y_test.reshape(-1).tolist()
x_train = x_train.tolist()
x_test = x_test.tolist()

'''
#svm train
linearModel = svm_train(y_train, x_train, '-t 0')
polynomialModel = svm_train(y_train, x_train,'-t 1')
rbfModel = svm_train(y_train, x_train,'-t 2')

f = open('svm_performance.txt', 'w')
p_labs, p_acc, p_vals = svm_predict(y_test, x_test, linearModel)
f.write(f'acc of linear kernel : {p_acc}\n')
p_labs, p_acc, p_vals = svm_predict(y_test, x_test, polynomialModel)
f.write(f'acc of polynomial kernel : {p_acc}\n')
p_labs, p_acc, p_vals = svm_predict(y_test, x_test, rbfModel)
f.write(f'acc of rbf kernel : {p_acc}\n')




#soft svm

grid = {'C':np.logspace(-3,2,5), 'gamma': np.logspace(-3,2,5), 'coef0':np.logspace(-3,2,4), 'degree':[1,2,3]}

#grid search linear
optParameter = grid_search(x_train,y_train,grid, 'softlinear')

softlinear = svm_train(y_train, x_train, '-t 0 -s 0 -c '+ optParameter['C'])


#grid search polynomial
optParameter = grid_search(x_train,y_train,grid, 'softpolynomial')

softpolynomial = svm_train(y_train, x_train, '-t 0 -s 0 -c '+ optParameter['C'] 
                       + ' -g ' + optParameter['gamma'] 
                       + ' -r ' + optParameter['coef0'])

#grid search rbf
optParameter = grid_search(x_train,y_train,grid, 'softrbf')

softrbf = svm_train(y_train, x_train, '-t 0 -s 0 -c '+ optParameter['C'] 
                       + ' -g ' + optParameter['gamma'])

p_labs, p_acc, p_vals = svm_predict(y_test, x_test, softpolynomial)
temp = optParameter['C']
print(f'opt parameter : {optParameter}')
print(f'acc of linear kernel : {p_acc}')
'''
grid = {'gamma':np.logspace(-7,1,1000)}

'''
#linear kernel + rbf kernel
x_train = np.array(x_train)
linearkernel = x_train @ x_train.T

#radial basis function: exp(-gamma*|u-v|^2),gamma : set gamma in kernel function (default 1/num_features)
dist = cdist(x_train, x_train, 'sqeuclidean')
rbfkernel = np.exp((1/numBit) * dist)
x_train = linearkernel+rbfkernel
idx = np.arange(len(x_train))+1
x_train = np.insert(x_train, 0, idx, axis=1)

combineModel = svm_train(y_train, x_train, '-t 4')


x_test = np.array(x_test)
linearkernel = x_test @ x_test.T
dist = cdist(x_test, x_test, 'sqeuclidean')
rbfkernel = np.exp((1/numBit) * dist)
x_test = linearkernel+rbfkernel
idx = np.arange(len(x_test))+1
x_test = np.insert(x_test, 0, idx, axis=1)

f = open('combinekernel.txt', 'w')
p_labs, p_acc, p_vals = svm_predict(y_test, x_test, combineModel)
f.write(f'acc of linear kernel + rbf kernel: {p_acc}\n')
'''


f = open('combinekernel.txt', 'w')

for i in grid['gamma']:
    #linear kernel + rbf kernel
    x = np.array(x_train).copy()
    linearkernel = x @ x.T

    #radial basis function: exp(-gamma*|u-v|^2),gamma : set gamma in kernel function (default 1/num_features)
    dist = cdist(x, x, 'sqeuclidean')
    rbfkernel = np.exp((i) * dist)
    x = linearkernel+rbfkernel
    idx = np.arange(len(x))+1
    x = np.insert(x, 0, idx, axis=1)

    combineModel = svm_train(y_train, x, '-t 4')


    x = np.array(x_test).copy()
    linearkernel = x @ x.T
    dist = cdist(x, x, 'sqeuclidean')
    rbfkernel = np.exp((i) * dist)
    x = linearkernel+rbfkernel
    idx = np.arange(len(x))+1
    x = np.insert(x, 0, idx, axis=1)

    
    p_labs, p_acc, p_vals = svm_predict(y_test, x, combineModel)
    f.write(f'gamma : {i}\n')
    f.write(f'acc of linear kernel + rbf kernel: {p_acc}\n')
    f.write(f'-------------------------------------------------\n')


