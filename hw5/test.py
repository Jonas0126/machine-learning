import numpy as np

x = np.zeros((3,2))
print(f'x = {x}')
t = np.arange(3)+1
x = np.insert(x, 0, t, axis=1)
print(f'x = {x}')