from gaussianGenerator import *
from polynomial import *
from sequentialEstimator import *
import numpy as np
import matplotlib.pyplot as plt


def draw(ax, line, n, mean, var, history_point, name):
    
    ax.set_title(name)
    ax.set_ylim(-20, 20)
    ax.set_xlim(-2, 2)

    mean_of_func = np.zeros(400)
    var_of_func_top = np.zeros(400)
    var_of_func_down = np.zeros(400)

    for i in range(len(line)):
        temp = np.zeros((1,n))
        for j in range(n):
            temp[0][j] = line[i]**j
        mean_of_func[i] = np.dot(temp, mean)
        var_of_func_top[i] = mean_of_func[i] + (a) + np.dot(temp, np.dot(var, temp.T))
        var_of_func_down[i] = mean_of_func[i] - (a) - np.dot(temp, np.dot(var, temp.T))

    if name == '10 incomes':
        num = 10
    elif name == '50 incomes':
        num = 50
    else:
        num = len(history_point)-1

    x = np.zeros(num)
    y = np.zeros(num)
    for i in range(0, num):
        x[i] = history_point[i+1][0]
        y[i] = history_point[i+1][1]
    ax.scatter(x,y, c='b') 
    ax.plot(line, mean_of_func, c='k')
    ax.plot(line, var_of_func_top, c='r')
    ax.plot(line, var_of_func_down, c='r')


def save_pic(history_point, num_param, w, a, n, name):

    line_x = np.linspace(-2, 2, 400)    
    fig = plt.figure(figsize=(10,7))
    ax = fig.add_subplot(221)
    ax.set_title('ground truth')
    ax.set_ylim(-20, 20)
    ax.set_xlim(-2, 2)
    y = np.zeros(400)
    var_top = np.zeros(400)
    var_down = np.zeros(400)
    for i in range(len(line_x)):
        for j in range(n):
            y[i] += w[j]*(line_x[i]**j)
            var_top[i] = y[i] + a
            var_down[i] = y[i] - a
    ax.plot(line_x, y, c='k')
    ax.plot(line_x, var_top, c='r')
    ax.plot(line_x, var_down, c='r')
    
    ax = fig.add_subplot(222)
    draw(ax, line_x, n, num_param['predict_mean'], num_param['predict_var'], history_point, 'predict result')


    ax = fig.add_subplot(223)
    draw(ax, line_x, n, num_param['10_mean'], num_param['10_var'], history_point, '10 incomes')

    ax = fig.add_subplot(224)
    draw(ax, line_x, n, num_param['50_mean'], num_param['50_var'], history_point, '50 incomes')
    plt.savefig(name+'.jpg')


def calculate_mean(x, x_mean,n, num):
    for i in range(n):
        x_mean[i][0] = update_mu(x_mean[i][0], x[i][0], num)

def add_newpoint(n, a, w):
    x, y = generator(n, a, w)
    temp = np.zeros((1,n))
    for i in range(n):
        temp[0][i] = x**i
    return temp, y

n = int(input('input n : '))
a = float(input('input a :'))
w = np.arange(1, n+1)
x_history = np.empty((n,1))



#initial prior w~N(0,b^-1 * I)
b = float(input('input initial prior N(0,b^-1 * I) : '))
prior_variance = np.zeros((n,n))
prior_mu = np.zeros((n,1))
posterior = np.zeros((n,n))
for i in range(n):
    prior_variance[i][i] = b


name = str(int(b))+str(n)+str(int(a))


num_param = {}
point_history = [[0, 0]]
num = 0


while(1):

    x, y = add_newpoint(n, a, w)
    temp = [[x[0][1], y]]
    point_history = np.append(point_history, temp, axis=0)
    num += 1

    

    #update prior
    #posterior mean =  inv(posterior_var)*([prior_var*prior_mu]+(1/a)*x_t*y)
    #inv(posterior var) = prior_var+(1/a)*xtx
    posterior_variance = prior_variance + (1/a) * np.dot(x.T, x)
    posterior_mean = np.dot(np.linalg.inv(posterior_variance),(np.dot(prior_variance, prior_mu) + (1/a) * (x.T) * y))

    print(f'Postirior Mean :\n{posterior_mean}\n\n')
    print(f'Posterior variance :\n{posterior_variance}\n')

    #predictive distribution
    pred_var = (a) + np.dot(x,np.dot(np.linalg.inv(posterior_variance), x.T))
    pred_mean = np.dot(x, posterior_mean)
    
    print(f'predictive distribution ~ N({pred_mean}, {pred_var})')

    if num == 10:
        num_param['10_mean'] = posterior_mean
        num_param['10_var'] = np.linalg.inv(posterior_variance)
    elif num == 50:
        num_param['50_mean'] = posterior_mean
        num_param['50_var'] = np.linalg.inv(posterior_variance)
        

    #check convarge
    convarge = 1
    for i in range(n):
        if abs(posterior_mean[i][0] - prior_mu[i][0]) > 0.000001:
            convarge = 0
         
    if convarge == 1:
        break
    

    #prior_variance = inv(posterior_variance)
    prior_mu = posterior_mean
    prior_variance = posterior_variance 

num_param['predict_mean'] = posterior_mean
num_param['predict_var'] = np.linalg.inv(posterior_variance)


save_pic(point_history, num_param, w, a, n, name)