import numpy as np

a = np.full((5,2,2), 2)
for i in range(5):
    a[i][0] = 1

a[2][0] = 4
print(a)
b = np.full((5,2), 3)
c = a[:,0,1].reshape(5,1)
x = np.zeros((3,3))
print(np.linalg.inv(x))
print(c.shape)
print(c.T @ c)