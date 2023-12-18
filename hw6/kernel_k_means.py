import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import imageio as iio
group = 2
img_size = 100
gamma_c = 0.00001
gamma_s = 0.00001

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


def kernel(x, x_):
    temp = np.exp(-gamma_s * S(x, x_))
    temp *= np.exp(-gamma_c * C(x, x_))

    return temp

#symmetric
def gram_matrix(data):
    M = np.zeros((img_size*img_size, img_size*img_size))
    for i in tqdm(range(img_size*img_size)):
        for j in range(i, img_size*img_size):
            M[i][j] = M[j][i] = kernel(data[i], data[j])
    return M



def initial(data, mode, M):
    indicator = np.zeros((group, img_size*img_size))
    num_cluster = np.zeros(group)

    #random
    if mode == 0:
        
        temp = np.arange(img_size*img_size)
        idx = np.random.choice(temp, size=group, replace=False)
        means = np.zeros((group, 5))
        for i in range(group):
            means[i] = data[idx[i]]
            indicator[i][idx[i]] = 1
        num_cluster = np.sum(indicator, axis=1)
    
    #k means++
    elif mode == 1:
        temp = np.arange(img_size*img_size)
        idx = np.random.choice(temp, size=1, replace=False)
        means = np.zeros((group, 5))
        means[0] = data[idx[0]]
        indicator[0][idx[0]] = 1
        min_D = np.ones(img_size*img_size) 

        #dist between means and point
        for i in range(1 ,group):
            prob = np.zeros(img_size*img_size)
            for j in range(img_size*img_size):
                means_idx = int(means[i-1][0]*img_size + means[i-1][1])
                
                D = M[j][j] - 2 * M[j][means_idx] + M[means_idx][means_idx]
                print(f'D = {D}, min_D[j] = {min_D[j]}')
                if D < min_D[j]:
                    min_D[j] = D
            print(f'min_D = {min_D}')
            min_D = min_D ** 2
            print(f'min_D = {min_D}')
            prob = min_D / np.sum(min_D)
            print(f'prob = {prob}')
            sum = 0
            for j in range(len(prob)):
                prob[j] += sum
                sum = prob[j]

            rand = np.random.rand()
            for j in range(len(prob)):
                if rand <= prob[j]:
                    means[i] = data[j]
                    indicator[i][j] = 1
                    break
        num_cluster = np.sum(indicator, axis=1)

    print(f'means = {means}')

    return indicator, num_cluster  

#calculate dist between k centers
def dist(num_cluster, indicator, M):
    d = np.zeros((group, img_size*img_size))
    
    for i in tqdm(range(img_size*img_size)):
        for j in range(group):
            d[j][i] = M[i][i]
    temp = indicator @ M.T
    
    #2/|C_k| * sum(1~n)a_kn*k(x_j, x_n)
    for i in range(group):
        temp[i] = temp[i] * (2/num_cluster[i])
    
    d = d - temp


    #a_kp*a_kq*k(x_p, x_q) => the diagonal element of a @ k @ a.t
    temp = indicator @ M @ indicator.T
    
    temp2 = np.ones((group, img_size*img_size))
    for i in range(group):
        temp2[i] = temp[i][i] * (1/(num_cluster[i]**2))
    
    d = d + temp2

    return d

def color_assign(indicator):
    out = np.zeros((100,100,3))
    for i in range(img_size):
        for j in range(img_size):
            for z in range(group):
                if indicator[z][i*img_size+j] == 1:
                    out[i][j] = color[z]
    return out

def kernel_k_means(data, path):
    M = gram_matrix(data)
    indicator, num_cluster = initial(data, 1, M)
    num = 0
    while 1:
        print(f'num = {num}')
        d = dist(num_cluster, indicator, M)
        indicator = np.zeros((group, img_size*img_size))
        pre_num = num_cluster.copy()
        for i in tqdm(range(img_size*img_size), dynamic_ncols=True):
            min = d[0][i]
            min_group = 0
            for j in range(1, group):
                if d[j][i] < min:
                    min = d[j][i]
                    min_group = j
            indicator[min_group][i] = 1
        num_cluster = np.sum(indicator, axis=1) 
        print(f'num_cluster : {num_cluster}')

        out = color_assign(indicator)
        

        plt.imsave(f'{path}output_{num}.png',out)
        dif = np.sum(np.abs(num_cluster-pre_num))
        print(f'num : {num}, dif : {dif}')
        if dif <= 10:
            break
        num += 1

color = [[1,0,0], [0,1,0], [0,0,1], [1,1,0], [0,0,0]]
color = np.array(color)
group = 2
imgidx = 1


for group in range(2,6):
    for imgidx in range(1,3):
        path = f'kernel_kmeans_img{imgidx}_cluster{group}_mode_1'

        if not os.path.isdir(path):
            os.mkdir(path)
        data = loaddata(f'image{imgidx}.png')
        kernel_k_means(data, f'{path}/')


        images = []
        imglist = os.listdir(path)

        for i in range(len(imglist)):
            images.append(iio.imread(f'{path}/output_{i}.png'))

        iio.mimsave(f'{path}/img{imgidx}_cluster{group}.gif', images, duration=1, loop=0)