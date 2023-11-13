import matplotlib.pyplot as plt
import matrixfunction as mf


def showreslut(method, y_fit, coefficient):
    loss = Loss(b, y_fit)

    plt.scatter(a, b)
    plt.plot(a,y_fit)
    plt.savefig(method+'.jpg')

    fittingline = str(coefficient[0][0]) + 'x^0'
    for i in range(1, len(coefficient)):
        fittingline += ' + ' + str(coefficient[i][0]) + 'x^' + str(i) 
    print(f'{method}:')
    print(f'{method} loss:{loss}')
    print(f'fitting line:{fittingline}\n')


def getdata(file, bases):
    f = open(file, mode = 'r')

    b = []
    A = []
    for line in f.readlines():
        line = line.strip('\n').split(',')
        temp = [1]
        for i in range(1, bases):
            temp.append(float(line[0]) ** i) 
        
        A.append(temp)        
        b.append([float(line[1])])
    return A, b

#計算fitting line
def cal_y_fit(x, y, coefficient):
    y_fit = [0] * len(x)
    for i in range(len(x)):
        for j in range(len(coefficient)):
            y_fit[i] += coefficient[j][0] * (x[i] ** j)
    return y_fit

def Loss(y, y_fit):
    loss = 0
    for i in range(len(y_fit)):
        loss += (b[i] - y_fit[i]) ** 2
    return loss




#(ATA + lambda*I)^-1 * AT * b
def rLSE(file, bases, weight):
    f = open(file, mode = 'r')

    b = []
    A = []
    for line in f.readlines():
        line = line.strip('\n').split(',')
        temp = [1]
        for i in range(1, bases):
            temp.append(float(line[0]) ** i) 
        
        A.append(temp)        
        b.append([float(line[1])])

    A_T = mf.transpose(A)
    ATA = mf.mul(A_T, A)

    #ATA + lamda*I
    for i in range(len(ATA)):
        ATA[i][i] += weight
    

    ATA_inverse = mf.inverse(ATA)

    ans = mf.mul(ATA_inverse, A_T)
    ans = mf.mul(ans, b)
    return ans


#newton' method : x_n+1 = x_n - f''(x_n)^-1 * f'(x_n)
#f(x) = |Ax - b|^2 -> g = f'(x) = 2ATAx-2ATb, Ht = f''(x) = 2ATA
#x_n+1 = x_n - (2ATA)^-1 * 2(ATA * x_n - ATb) 
def Newton(file, bases):
    
    f = open(file, mode = 'r')

    x_0 = [[0]*1 for _ in range(bases)]
    b = []
    A = []
    for line in f.readlines():
        line = line.strip('\n').split(',')
        temp = [1]
        for i in range(1, bases):
            temp.append(float(line[0]) ** i) 
        
        A.append(temp)        
        b.append([float(line[1])])

    A_T = mf.transpose(A)
    ATA = mf.mul(A_T, A)

    Ht = mf.inverse(ATA)

    #f'(x)
    g = mf.mul(ATA, x_0)

    temp = mf.mul(A_T, b)
    for i in range(len(g)):
        for j in range(len(g[0])):
            g[i][j] -= temp[i][j]

    ans = mf.mul(Ht, g)

    for i in range(len(x_0)):
        for j in range(len(x_0[0])):
            ans[i][j] = x_0[i][j] - ans[i][j]
        
    return ans


#f(x) = |Ax - b|^2 + lambda|x|
#f'(x) = 2(ATAx - ATb) + lambda
#x_1 = x_0 - L * f'(x_0)
def steepnest(file, basis, weight):
    A, b = getdata(file, basis)

    #learning rate
    L = 0.000000088
    
    #x_0
    x = [[0]*1 for _ in range(basis)]
    #x_1
    x_n = [[0]*1 for _ in range(basis)]
    
    A_T = mf.transpose(A)
    ATA = mf.mul(A_T,A)

    #AT_b = AT*b
    AT_b = mf.mul(A_T,b)

    for _ in range(1000000):
        #ATA * x
        temp = mf.mul(ATA, x)

 
        #2((ATA*x)-AT*b) + lambda
        for i in range(len(temp)):
            if(x[i][0] >= 0):
                temp[i][0] = L * ((2 * (temp[i][0] - AT_b[i][0])) + weight)
            else:
                temp[i][0] = L * ((2 * (temp[i][0] - AT_b[i][0])) - weight)
        #x_1 = x_0 - f'(x_0)
        for i in range(len(x)):
            x_n[i][0] = x[i][0] - temp[i][0]

        x = x_n.copy()

    return  x



filename = 'test.txt'
f = open(filename, mode = 'r')
a = []
b = []
for line in f.readlines():
    line = line.strip('\n').split(',')
    a.append(float(line[0]))
    b.append(float(line[1]))



rlse = rLSE(filename, 3, 10000)
y_fit = cal_y_fit(a, b, rlse)
showreslut('rlse', y_fit, rlse) 

newton = Newton(filename, 3)
y_fit = cal_y_fit(a, b, newton)
showreslut('newton', y_fit, newton)

steep = steepnest(filename, 3, 10000)
y_fit = cal_y_fit(a, b, steep)
showreslut('steep', y_fit, steep)