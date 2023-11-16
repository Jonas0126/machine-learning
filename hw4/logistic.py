from gaussianGenerator import *
import matplotlib.pyplot as plt
import numpy as np

def update(X, w, Y, flag):
    #f(x) = 1 / (1 + e^(wT*X))
    f_x = 1 / (1+(np.exp(np.dot(X, -1*w))))
    #gradient = XT(f(x) - Y)
    gradient = np.dot(X.transpose(),(f_x - Y))

    if flag == 0:
        w = w - (gradient * 0.005)

    else:
        #e = e^-(wT*X)
        D = np.zeros((len(X),len(X)))
        print(np.shape(D))
        for i in range(len(D)):
            e = np.dot(X[i], w)
            D[i][i] = np.exp(-e) / ((1+np.exp(e))**2)
        print(D)
        
        
        hassian = np.dot(X.transpose(), np.dot(D, X))
    
        try:
            hassian_inv = np.linalg.inv(hassian)
            print(f'hassian_inv = {hassian_inv}')
            print(f'hassian_inv @ gradient = {hassian_inv @ gradient}')
            w -= np.dot(hassian_inv, gradient) * 0.001
        except:
            w = w - (gradient * 0.005)
    return w

def result(w, alldata, name):
    TP = 0
    FP = 0
    FN = 0
    TN = 0
    for i in range(len(alldata[0])):
        x = np.zeros((3, 1))
        for j in range(3):
            x[j] = alldata[0][i] ** j

        f_x = 1 / (1 + np.exp(np.dot(x.transpose(), -1 * w)))
        
        if f_x >= 0.5:
            plt.scatter(alldata[0][i], alldata[1][i],c = 'b')
            if i < 50:
                TP += 1
            else:
                FN += 1
        else:
            plt.scatter(alldata[0][i], alldata[1][i],c = 'r')
            if i >= 50:
                TN += 1
            else:
                FP += 1  
        plt.savefig(f'{name}.jpg')
    return TP, FP, FN, TN

mu = np.zeros(4)
var = np.zeros(4)

for i in range(4):
    mu[i] = float(input(f'mu{i+1} = '))
    var[i] = float(input(f'var{i+1} = '))

n = int(input('input n = '))

d1 = np.zeros((2,n))
d2 = np.zeros((2,n))

for i in range(n):
    d1[0][i] = gaussian(mu[0], var[0])
    d1[1][i] = gaussian(mu[1], var[1])
    d2[0][i] = gaussian(mu[2], var[2])
    d2[1][i] = gaussian(mu[3], var[3])

plt.scatter(d1[0],d1[1],c='b')
plt.scatter(d2[0], d2[1], c='r')

plt.savefig('groundtruth_case2.jpg')


#alldata[0 = x | 1 = y][data]
alldata = np.append(d1, d2, axis=1)
print(alldata)


#inttial w = [1, 1, 1]^T
w = np.ones((3,1))





#X => [number of data][3] nx3
X = np.zeros((len(alldata[0]), 3))
#Y => [number of data][1] nx1
Y = np.zeros((len(alldata[1]),1))
for j in range(len(alldata[0])):
    for i in range(3):
        X[j][i] = alldata[0][j] ** i
    if j < 50:
        Y[j][0] = 1
    else:
        Y[j][0] = 0

print(Y)


count = 0
while(1):
    w1 = update(X,w,Y, 0)
    print(f'w1 = {w1}')
    #
    if abs(w1[0] - w[0]) < 0.001 and abs(w1[1] - w[1]) < 0.001 and abs(w1[2] - w[2]) < 0.001:
        break
    
    w = w1
    
    count += 1
    if(count % 50000 == 0):
        result(w,alldata,'gradient_case2')
    if(count == 200000):
        break
    

TP, FP, FN, TN = result(w,alldata,'gradient_case2')
f = open('case2.txt', mode = 'w')
f.write('gradient descent: \n')
f.write(f'w:\n{w}\n')

#sensitivity = TP /(TP+FN)
#specificity = TN /(TN+FP)
f.write('Confusion Matrix:\n')
f.write('               predict cluster 1 predict cluster 2\n')
f.write(f'is cluster 1          {TP}             {FP}\n')
f.write(f'is cluster 2          {FN}             {TN}\n')
f.write(f'Sensitivity (successfully predict cluster 1) : {TP/(TP+FN)}\n')
f.write(f'Specificity (successfully predict cluster 2) : {TN/(TN+FP)}\n')
f.write('----------------------------------------\n')


#inttial w = [0, 0, 0]^T
w = np.zeros((3,1))


while(1):
    w1 = update(X,w,Y, 1)
    print(f'w1 = {w1}')
    if abs(w1[0] - w[0]) < 0.001 and abs(w1[1] - w[1]) < 0.001 and abs(w1[2] - w[2]) < 0.001:
        break
    
    w = w1

    
    count += 1
    if(count % 50000 == 0):
        result(w,alldata,'gradient_case2')
    if(count == 200000):
        break
    

TP, FP, FN, TN = result(w,alldata,'newton_case2')
f.write('newton\'s method: \n')
f.write(f'w:{w}\n')

f.write('Confusion Matrix:\n')
f.write('               predict cluster 1 predict cluster 2\n')
f.write(f'is cluster 1          {TP}             {FP}\n')
f.write(f'is cluster 2          {FN}             {TN}\n')
f.write(f'Sensitivity (successfully predict cluster 1) : {TP/(TP+FN)}\n')
f.write(f'Specificity (successfully predict cluster 2) : {TN/(TN+FP)}\n')
f.write('----------------------------------------')
f.close()