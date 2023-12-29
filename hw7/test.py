import os
import numpy as np
from matplotlib import pyplot as plt
from tqdm import trange

sub_num = 15
img_num = 11
train_num = 9
test_num = 2

height, width = 231, 195
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

def LDA(data, k=25, S=1):
    mean = np.mean(data, axis=0)
    # Sw, Sb
    Sw = np.zeros((len(data[0]), len(data[0])), dtype=np.float32)
    Sb = np.zeros((len(data[0]), len(data[0])), dtype=np.float32)
    for sub in trange(sub_num):
        xi = data[sub * train_num : (sub + 1) * train_num]
        mj = np.mean(xi, axis=0)
        Sw += (xi - mj).T @ (xi - mj)
        Sb += len(xi) * (mj - mean).reshape(-1, 1) @ (mj - mean).reshape(1, -1)
    
    # Pseudo inv.
    eigenvalue, eigenvector = np.linalg.eig(np.linalg.pinv(Sw) @ Sb)
    
    for i in range(len(eigenvector[0])):
        eigenvector[:,i] = eigenvector[:,i] / np.linalg.norm(eigenvector[:,i])
    eigenindex = np.argsort(-eigenvalue)
    eigenvector = eigenvector[:, eigenindex]
    W = eigenvector[:, :k].real

    return W, mean

def imageCompression(data, S):
    d = np.zeros((len(data), height//S, width//S))
    for n in range(len(data)):
        d[n] = np.full((height//S, width//S), np.mean(data[n]))
        img = data[n].reshape(height, width)
        for i in range(0, height - S + 1, S):
            for j in range(0, width - S + 1, S):
                tmp = 0
                # Summation SxS area in original image
                for r in range(S):
                    for c in range(S):
                        tmp += img[i + r][j + c]
                # New value is the avg. value of SxS area in original image
                d[n][i//S][j//S] = tmp // (S**2)
    return d.reshape(len(data),-1)



def PCA(data, k=25):
    mean = np.mean(data, axis=0)
    cov = (data - mean) @ (data - mean).T
    eigenvalue, eigenvector = np.linalg.eig(cov)
    eigenvector = data.T @ eigenvector
    
    # Normalize w
    for i in range(len(eigenvector[0])):
        eigenvector[:,i] = eigenvector[:,i] / np.linalg.norm(eigenvector[:,i])
        
    # Seclect first k largest eigenvalues
    eigenindex = np.argsort(-eigenvalue)
    eigenvector = eigenvector[:, eigenindex]
    
    W = eigenvector[:, :k].real
    
    return W, mean

def eigenFace(W, file_path, k=25, S=1):
    fig = plt.figure()
    for i in range(k):
        img = W[:,i].reshape(height//S, width//S)
        plt.imshow(img, cmap='gray')
        fig.savefig(f'eigenface_{i:02d}.jpg')

def reconstructFace(W, mean, data, file_path, S=1):
    if mean is None:
        mean = np.zeros(W.shape[0])
    
    sel = np.random.choice(sub_num * train_num, 10, replace=False)
    img = []
    for i in sel:
        x = data[i].reshape(1, -1)
        reconstruct = (x - mean) @ W @ W.T + mean
        img.append(reconstruct.reshape(height//S, width//S))
        
        plt.imsave(f're_{i:02d}.jpg',reconstruct.reshape(height//S, width//S), cmap='gray') # Reconstruct face
        



train_data, test_data = readData()
PCA_file = './Experiment Result/PCA_LDA/PCA/'
W_PCA, mean_PCA = PCA(train_data, k=25)
print(f'w_pca shape = {W_PCA.shape}')
eigenFace(W_PCA, PCA_file + 'eigenfaces/', k=25)
reconstructFace(W_PCA,mean_PCA,train_data,'11')
scalar = 3  # 45000 x (1/9) -> 5000
LDA_file = './Experiment Result/PCA_LDA/LDA/'
data = imageCompression(train_data, scalar)

'''     
W_LDA, mean_LDA = LDA(data, k=25, S=scalar)
eigenFace(W_LDA, LDA_file + 'fisherfaces/', k=25, S=scalar)
'''