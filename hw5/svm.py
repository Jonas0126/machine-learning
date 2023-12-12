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
        f.close()
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
        f.close()
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
        f.close()
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
print(f'acc :{p_acc[0]}, mse : {p_acc[1]}')
f.write(f'acc of rbf kernel : {p_acc}\n')



#soft svm

grid = {'C':np.logspace(-3,2,5), 'gamma': np.logspace(-3,2,5), 'coef0':np.logspace(-3,2,4), 'degree':[1,2,3]}

#grid search linear
optParameter = grid_search(x_train,y_train,grid, 'softlinear')

softlinear = svm_train(y_train, x_train, '-t 0 -s 0 -c '+ optParameter['C'])
p_labs, p_acc, p_vals = svm_predict(y_test, x_test, softlinear)
f = open('softlinear.txt', 'a')
f.write(f'opt parameter : {optParameter}\n')
f.write(f'acc of linear kernel : {p_acc}\n')

#grid search polynomial
optParameter = grid_search(x_train,y_train,grid, 'softpolynomial')

softpolynomial = svm_train(y_train, x_train, '-t 0 -s 0 -c '+ optParameter['C'] 
                       + ' -g ' + optParameter['gamma'] 
                       + ' -r ' + optParameter['coef0'])

p_labs, p_acc, p_vals = svm_predict(y_test, x_test, softpolynomial)
f = open('softpolynomial.txt', 'a')
f.write(f'opt parameter : {optParameter}\n')
f.write(f'acc of polynomial kernel : {p_acc}\n')



#grid search rbf
optParameter = grid_search(x_train,y_train,grid, 'softrbf')

softrbf = svm_train(y_train, x_train, '-t 0 -s 0 -c '+ optParameter['C'] 
                       + ' -g ' + optParameter['gamma'])

p_labs, p_acc, p_vals = svm_predict(y_test, x_test, softrbf)
f = open('softrbf.txt', 'a')
f.write(f'opt parameter : {optParameter}\n')
f.write(f'acc of rbf kernel : {p_acc}\n')


'''


f = open('combinekernel2222.txt', 'w')
best = 0
opt = 0

#linear kernel + rbf kernel
x = np.array(x_train).copy()
linearkernel = x @ x.T

#radial basis function: exp(-gamma*|u-v|^2),gamma : set gamma in kernel function (default 1/num_features)
dist = cdist(x, x, 'sqeuclidean')
rbfkernel = np.exp(-0.1 * dist)
x = linearkernel+rbfkernel
idx = np.arange(len(x))+1
x = np.insert(x, 0, idx, axis=1)

combineModel = svm_train(y_train, x, '-t 4')


x = np.array(x_test).copy()
linearkernel = x_test @ np.array(x_train).T
dist = cdist(x_test, np.array(x_train), 'sqeuclidean')
rbfkernel = np.exp(-0.1 * dist)
x = linearkernel+rbfkernel
idx = np.arange(len(x))+1
x = np.insert(x, 0, idx, axis=1)

    
p_labs, p_acc, p_vals = svm_predict(y_test, x, combineModel)

f.write(f'gamma : -0.1\n')
f.write(f'acc of linear kernel + rbf kernel: {p_acc}\n')


