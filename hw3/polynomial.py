from gaussianGenerator import *
import matplotlib.pyplot as plt
from scipy import stats

def generator(n,a,w):
    x = np.random.uniform(-1,1,1)
    y = 0
    e = gaussian(0,a)
    for i in range(n):
        y += w[i]*(x[0]**i)
    
    y+=e
    return x[0], y

if __name__ == '__main__':

    n = int(input('input n : '))
    a = float(input('input a :'))
    w = [0] * n
    for i in range(n):
        w[i] = float(input(f'input w{i} : '))

    x, y = generator(n, a, w)
    print(f'x = {x}, y = {y}')


'''
x = [0] * 100
y = [0] * 100
for i in range(100):
    x[i], y[i] = generator(2, 1, w)

print(x)
print(y)
print(stats.kstest(x, 'norm').pvalue)
plt.scatter(x,y)
plt.savefig('01.jpg')
'''