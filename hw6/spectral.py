import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
group = 2
img_size = 100
gamma_c = 0.01
gamma_s = 0.01


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

def degree_matrix(W):
    D = np.zeros((img_size*img_size, img_size*img_size))
    for i in tqdm(range(img_size*img_size)):
        D[i][i] = np.sum(W[i]) - W[i][i]
    return D


def spctral(data, path):
    W = gram_matrix(data)
    D = degree_matrix(W)
    print(f'W : {W}')
    print(f'W.shape : {W.shape}')
    print(f'D : {D}')
    print(f'D.shape : {D.shape}')
    L = D - W
    print(f'L : {L}')
    print(f'L.shape : {L.shape}')
    eigenvalue, eigenvector = np.linalg.eig(L)
    eigenvalue_idx = np.argsort(eigenvalue)
    print(f'eigenvalue index : {eigenvalue_idx}')
    print(f'eigenvector : ', end='')
    for i in range(10):
        print(f'{eigenvector[eigenvalue_idx[i]]}')

imgidx = 1
group = 2
data = loaddata(f'image{imgidx}.png')
path = f'spectral_img{imgidx}_cluster{group}'
spctral(data, path)