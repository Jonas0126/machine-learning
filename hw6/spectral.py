import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import imageio as iio
img_size = 100
gamma_c = 1 / (256 * 256)
gamma_s = 0.0001


def loaddata(file_name):
    img = cv2.imread(file_name, cv2.IMREAD_COLOR)
    data = np.zeros((img_size * img_size, 5))
    print(f'img[0][0] = {img[0][0]}')
    for i in range(img_size):
        for j in range(img_size):
            data[i*img_size+j][0] = i
            data[i*img_size+j][1] = j
            for k in range(2, 5):
                data[i*img_size+j][k] = img[i][j][k-2]
    return data


#calculate ||S(x)-S(x')||^2
def S(x, x_):
    temp = 0
    for i in range(2):
        temp += (x[i]-x_[i]) ** 2
    
    return temp

#calculate ||C(x)-C(x')||^2
def C(x, x_):
    temp = 0
    for i in range(2, 5):
        temp += (x[i]-x_[i]) ** 2

    return temp

#kernel method
def kernel(x, x_):
    temp = np.exp(-gamma_s * S(x, x_))
    temp *= np.exp(-gamma_c * C(x, x_))

    return temp


#gram matrix is symmetric
def gram_matrix(data):
    M = np.zeros((img_size*img_size, img_size*img_size))
    for i in tqdm(range(img_size*img_size)):
        for j in range(i, img_size*img_size):
            M[i][j] = M[j][i] = kernel(data[i], data[j])
    return M

#calculate degree matrix
def degree_matrix(W):
    D = np.zeros((img_size*img_size, img_size*img_size))
    for i in tqdm(range(img_size*img_size)):
        D[i][i] = np.sum(W[i])
    return D

#spectral clustering
def spctral(data, path, mode):
    '''
    fw = open('testw1.txt', 'w')
    fd = open('testd1.txt', 'w')
    fl = open('testl1.txt', 'w')
    #calculate weighted adjacency matrix and degree matrix
    W = gram_matrix(data)
    D = degree_matrix(W)
        
    #compute the unnormalized Laplacian L     
    L = D - W
    for i in range(img_size*img_size):
        fw.write(f'{W[i]}\n')
        fd.write(f'{D[i]}\n')
        fl.write(f'{L[i]}\n')
    
    
    #normalized
    norm_D = np.sqrt(D)
    for i in range(len(D)):
        norm_D[i][i] = 1 / norm_D[i][i] 
    L = norm_D @ L @ norm_D
    
    
    #compute the first k eigenvectors of L => U
    eigenvalue, eigenvector = np.linalg.eig(L)
    eigenvalue_idx = np.argsort(eigenvalue)

    #U contain the eigenvectors of L, dim = n*group
    U = []
    for i in range(1, group+1):
        U.append(eigenvector[:,eigenvalue_idx[i]])
    U = (np.array(U)).T
    np.save(f'norm_eigenvalue_img{imgidx}', eigenvalue)    
    np.save(f'norm_eigenvector_img{imgidx}', eigenvector)
    '''

    
    eigenvalue_img1 = np.load('eigenvalue_img1.npy')
    eigenvector_img1 = np.load('eigenvector_img1.npy')
    eigenvalue_idx = np.argsort(eigenvalue_img1)
    U = []
    for i in range(1, group+1):
        U.append(eigenvector_img1[:,eigenvalue_idx[i]].real)
    U = (np.array(U)).T 
    k_means(U, path, mode)
    

def initial(data, mode):
    
    #random
    if mode == 0:
        
        temp = np.arange(img_size*img_size)
        idx = np.random.choice(temp, size=group, replace=False)
        means = np.zeros((group, group))

        for i in range(len(idx)):
            means[i] = data[idx[i]]
        
    elif mode == 1:
        temp = np.arange(img_size*img_size)
        idx = np.random.choice(temp, size=1, replace=False)
        means = np.zeros((group, group))
        means[0] = data[idx[0]]
        
        min_D = np.ones(img_size*img_size) 
        min_D *= 10000
        #dist between means and point
        for i in range(1 ,group):
            prob = np.zeros(img_size*img_size)
            for j in range(img_size*img_size):
                
                
                D = np.sum((data[j] - means[i-1])**2) 
                min_D[j] = min(min_D[j], D ** 0.5)
            min_D = min_D ** 2
            prob = min_D / np.sum(min_D)           

            #select next mean point        
            means_idx = np.random.choice(temp, size = 1, p = prob).item()
            means[i] = data[means_idx]

        print(f'means : {means}')
    return means

#calculate the distance between two points
def dist(x, x_):
    d = 0
    for i in range(len(x)):
        d += (x[i] - x_[i]) ** 2
    return d

def update_indicator(data, means):
    ind = np.zeros((group, img_size*img_size))

    for i in range(img_size*img_size):
        min = dist(data[i], means[0])
        group_idx = 0
        for j in range(1, group):
            temp = dist(data[i], means[j])
            if temp < min:
                min = temp
                group_idx = j

        ind[group_idx][i] = 1

    return ind


def color_assign(indicator):
    
    out = np.zeros((100,100,3))
    for i in range(img_size):
        for j in range(img_size):
            for z in range(group):
                if indicator[z][i*img_size+j] == 1:
                    out[i][j] = color[z]
    return out

def drawEigenspace(indicator, num_cluster, data):
    eigenspace = []
    
    for i in range(group):
        temp = np.zeros((2, int(num_cluster[i])))
        num = 0
        for j in range(img_size*img_size):
            if indicator[i][j] == 1:
                temp[0][num] = data[j][0]
                temp[1][num] = data[j][1]
                num += 1
        eigenspace.append(temp)
    plt.scatter(eigenspace[0][0], eigenspace[0][1], color = 'b')
    plt.scatter(eigenspace[1][0], eigenspace[1][1], color = 'r')
    plt.show()
    plt.savefig('eigen.png')
def k_means(data, path, mode):

    #initial center
    pre_means = initial(data, mode)
  
    print(f'data : {data}')
    num = 0
    
    
    num_cluster = np.zeros(group)
    while 1:
        print(f'num : {num}')
        print(f'pre means : {pre_means}')
        pre_num = num_cluster.copy()
        indicator = update_indicator(data, pre_means)
        print(f'shape of indicator : {indicator.shape}')
        
        
        num_cluster = np.sum(indicator, axis=1)
        print(f'num of cluster : {num_cluster}, sum : {np.sum(num_cluster)}')


        means = np.zeros((group, group))
        for i in range(group):            
            for j in range(img_size*img_size):
                if indicator[i][j] == 1:
                    means[i] += data[j]
                    
            means[i] = means[i] / num_cluster[i]


        out = color_assign(indicator)

        #plt.imsave(f'{path}output_{num}.png',out)
        drawEigenspace(indicator, num_cluster, data)
        
        dif = np.sum(np.abs(num_cluster-pre_num))
        print(f'num : {num}, dif : {dif}')
        if dif <= 4:
            break
        
        pre_means = means.copy()
        num += 1

imgidx = 1
group = 2
color = [[1,0,0], [0,1,0], [0,0,1], [1,1,0], [0,0,0]]
color = np.array(color)



for group in range(2,3):
    for imgidx in range(1,2):
        for mode in range(0,1):
            path = f'spectral_img{imgidx}_cluster{group}_mode_{mode}'

            if not os.path.isdir(path):
                os.mkdir(path)
            data = loaddata(f'image{imgidx}.png')
            spctral(data, f'{path}/', mode)


            #images = []
            #imglist = os.listdir(path)

            #for i in range(len(imglist)):
                #images.append(iio.imread(f'{path}/output_{i}.png'))

            #iio.mimsave(f'{path}/img{imgidx}_cluster{group}.gif', images, duration=1, loop=0)