import os
import numpy as np
from matplotlib import pyplot as plt
from tqdm import trange


sub_num = 15
img_num = 11
k = 25
width = 195
height = 231
subject = ['centerlight', 'glasses', 'happy', 'leftlight',
            'noglasses', 'normal', 'rightlight', 'sad',
            'sleepy', 'surprised', 'wink']

def readimg(file):

    with open(file, 'rb') as f:
        
        f.readline() 
        f.readline()
        width, height = [int(i) for i in f.readline().split()]
        #print(f'width : {width}, height : {height}')
        img = np.zeros((height, width))
        for r in range(height):
            for c in range(width):
                img[r][c] = ord(f.read(1))

        return img.reshape(-1)

def readData():
    train = []
    for i in range(sub_num):
        for j in range(img_num):
            file = f"Yale_Face_Database/Training/subject{i+1:02d}.{subject[j]}.pgm"
            if os.path.isfile(file):
                train.append(readimg(file))

    test = []
    for i in range(sub_num):
        for j in range(img_num):
            file = f'Yale_Face_Database/Testing/subject{i+1:02d}.{subject[j]}.pgm'
            if os.path.isfile(file):
                train.append(readimg(file))
    return np.array(train), np.array(test)


def pca(M):
    mu = np.mean(M, axis=0)
    #M = M-mu
    cov = (M-mu) @ (M-mu).T
    eigenvalue, eigenvector = np.linalg.eig(cov)
    print(eigenvector.shape)
    for i in range(len(eigenvector[0])):
        eigenvector[:,i] = eigenvector[:,i] / np.linalg.norm(eigenvector[:,i])
    eigenvalue_idx = np.argsort(-eigenvalue)
    U = []
    for i in range(0, k):
        U.append(eigenvector[:,eigenvalue_idx[i]].real)

    return np.array(U), mu

def eigenface(eigenvector, data, file):
    eigenface = eigenvector@data
    for i in range(len(eigenvector)):
        img = np.zeros((height, width))
        for j in range(height):
            for r in range(width):
                img[j][r] = eigenface[i][j*width+r]
        plt.imsave(f'{file}/eigenface_{i}.jpg', img, cmap='gray')

def imageCompression(data, S):
    d = np.zeros((len(data), height//S, width//S))
    for n in range(len(data)):
        d[n] = np.zeros((height//S, width//S))
        img = data[n].reshape(height, width)
        for i in range(0, height - S + 1, S):
            for j in range(0, width - S + 1, S):
                tmp = 0

                for k in range(S):
                    for r in range(S):
                        tmp += img[i + k][j + r]

                d[n][i//S][j//S] = tmp // (S**2)
    return d.reshape(len(data),-1)


def eigenface_lda(eigenvector, file):
    eigenvector = eigenvector.T
    new_height = height // 3
    new_width = width // 3
    for i in range(k):
        img = np.zeros((new_height, new_width))
        for j in range(new_height):
            for r in range(new_width):
                img[j][r] = eigenvector[i][j*new_width+r]
        plt.imsave(f'{file}/eigenface_{i}.jpg', img, cmap='gray')

def lda(M):
    mu = np.mean(M, axis=0)

    sw = np.zeros((len(M[0]), len(M[0])))
    sb = np.zeros((len(M[0]), len(M[0])))
    for sub in trange(sub_num):
        xi = M[sub * 9 : (sub + 1) * 9]
        mj = np.mean(xi, axis=0)
        sw += (xi - mj).T @ (xi - mj)
        sb += len(xi) * (mj-mu).T @ (mj-mu)

    eigenvalue, eigenvector = np.linalg.eig(np.linalg.pinv(sw) @ sb)
    for i in range(len(eigenvector[0])):
        eigenvector[:,i] = eigenvector[:,i] / np.linalg.norm(eigenvector[:,i])
    eigenvalue_idx = np.argsort(-eigenvalue)

    eigenvector = eigenvector[:, eigenvalue_idx]
    U = eigenvector[:, :k].real

    return U, mu


train, test = readData()
print(train.shape)
eigenvector, mu = pca(train)
print(eigenvector.shape)
print(mu.shape)
eigenface(eigenvector, train, 'pca_eigenface/')
data = imageCompression(train,3)
eigenvector, mu = lda(data)
print(eigenvector.shape)
eigenface_lda(eigenvector, 'lda_eigenface/')

print(mu.shape)







