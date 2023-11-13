import numpy as np
from scipy import stats
#cos(2*pi*a)*(-2*ln(b))^(1/2)
def boxMuller():
    a = np.random.uniform(0,1,1)
    b = np.random.uniform(0,1,1)
    return np.cos(2*np.pi*a[0])*((-2*np.log(b[0]))**(1/2))


def gaussian(mean, variance):
    point = boxMuller()
    return (variance**(1/2))*point+mean


if __name__ == '__main__':
    mean = float(input('mean : '))
    variance = float(input('variance : '))
    print(gaussian(mean, variance))
    
    