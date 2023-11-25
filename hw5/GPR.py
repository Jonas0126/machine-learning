import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.optimize import minimize



def loadData():
    x = []
    y = []
    with open("data/input.data", 'r') as file:
        for line in file.readlines():
            d = line.split()
            x.append(float(d[0]))
            y.append(float(d[1]))
    x = np.array(x)
    y = np.array(y)
    return x, y
#rational quadratic kernel
#K(x,x') = (1+((x-x')^2)/(2*a*l^2))^-a
def kernel(x,x_,a,l):
    
    dist = cdist(x.reshape(-1,1), x_.reshape(-1,1), 'sqeuclidean')
    return (1 + (dist)/(2*a*(l*l)))**(-a)

def predict(x, x_, y, a, l):
    kxx = kernel(x, x, a, l)
    kxx_ = kernel(x, x_, a, l)
    kx_x_ = kernel(x_, x_, a, l)
    mu = kxx_.T @ np.linalg.inv(kxx+b*np.identity(len(x))) @ y
    temp = kx_x_ - (kxx_.T @ np.linalg.inv(kxx+b*np.identity(len(x))) @ kxx_)
    var = np.zeros(len(x_))
    for i in range(len(x_)):
        var[i] = temp[i][i]
    return mu, var

def drawLine(x, x_, y, mu, var):
    top = mu + 2 * var
    down = mu - 2 * var


    plt.figure(figsize=(16,10))
    plt.xlim(-60, 60)
    
    
    plt.plot(x, y, 'o', color = 'r')
    plt.plot(x_.reshape(-1), mu.reshape(-1), '-', color = 'b')
    plt.fill_between(x_, top, down, color = 'pink')

    plt.show()
    
    
def logLikelihood(par, x, y):
    par = par.reshape(len(par), 1)
    K = kernel(x,x,par[0],par[1]) + b * np.identity(len(x))
    print(f'K = {K}')
    L = 0.5 * ((y.T @ np.linalg.inv(K) @ y) + np.log(np.linalg.det(K)) + (len(x) * np.log(2*np.pi)))
    
    return L

#initial
a = 1
l = 1
b = 1/5
v = 1

x, y = loadData()
x = np.array(x)
y = np.array(y)
x_ = np.linspace(-60, 60, 500)


mu, var = predict(x,x_,y,a,l)
drawLine(x, x_, y, mu,var)

bestPar = minimize(logLikelihood, [a, l], bounds=((1e-5,1e5), (1e-5, 1e5)), args=(x, y))
a = bestPar.x[0]
l = bestPar.x[1]
print(f'l = {l}')
mu, var = predict(x,x_,y,a,l)
drawLine(x, x_, y, mu,var)