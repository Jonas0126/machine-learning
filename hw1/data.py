import numpy as np
import matplotlib.pyplot as plt
x = np.arange(-6, 6, 0.2)

y = 2 * (x**3) - 4 * x + 6 
noise = np.random.uniform(-3,3,len(x))
y += noise
print(x)
print(y)
plt.scatter(x,y)
plt.savefig('data.jpg')
f = open('test1.txt', 'w')
for i in range(len(x)):
    print(f'{x[i]},{y[i]}', file=f)