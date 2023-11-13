import matplotlib.pyplot as plt
import math


def to_int(B):
    return int.from_bytes(B, byteorder='big')

def read_label(file):
    with open(file, 'rb') as f:
        magicnum = to_int(f.read(4))
        #print(f'magic num = {magicnum}')
        labelnum = to_int(f.read(4))
        #print(f'label num = {labelnum}')
        labellist = [0] * labelnum
        for i in range(labelnum):
            labellist[i] = to_int(f.read(1))
    return labellist

def read_img(file):
    with open(file, 'rb') as f:
        magicnum = to_int(f.read(4))
        #print(f'magic num = {magicnum}')
        imgnum = to_int(f.read(4))
        #print(f'img num = {imgnum}')
        row = to_int(f.read(4))
        #print(f'row = {row}')
        col = to_int(f.read(4))
        #print(f'column = {col}')
        imglist = [[0 for i in range(col*row)] for i in range(imgnum)]
        for i in range(imgnum):
            for j in range(row*col):
                    box = int((to_int(f.read(1)) / 8))

                    imglist[i][j] = box
    return imglist

#計算label x(x=0~9)在(i,j)有多少bin y(y=0~31)
def calculate_numbin(traindata, trainlabel):
    numbin = [[[0] * 32 for _ in range(784)] for _ in range(10)]

    
    for i in range(784):
        for k in range(60000):
            numbin[trainlabel[k]][i][traindata[k][i]] += 1
    for i in range(10):
        for j in range(784):
            for k in range(32):
                if numbin[i][j][k] == 0:
                    numbin[i][j][k] = 1
    
    return numbin
#計算label x(x=0~9)有幾個
def calculate_numlabel(trainlabel):
    numlabel = [0] * 10
    for i in range(len(trainlabel)):
        numlabel[trainlabel[i]] += 1
    return numlabel


#P(pixel[1]|0).....P(pixel[784]|0)
#~~~~
#P(pixel[1]|9).....P(pixel[784]|9)
def calculate_prior(numlabel, numbin):
    prior = [[[0]*32 for _ in range(784)] for _ in range(10)]
    for i in range(10):
        for j in range(784):
            for k in range(32):
                prior[i][j][k] = numbin[i][j][k] / numlabel[i]
    return prior

def calculate_mean(traindata, trainlabel, numlabel):
    mean = [[0] * 784 for _ in range(10)]

    for i in range(784):
        for j in range(len(trainlabel)):
            mean[trainlabel[j]][i] += traindata[j][i]
    for i in range(10):
        for j in range(784):
            mean[i][j] /= numlabel[i]
    return mean

def calculate_variance(traindata, trainlabel, mean, numlabel):
    variance = [[0] * 784 for _ in range(10)]

    for i in range(784):
        for j in range(len(trainlabel)):
            variance[trainlabel[j]][i] += (traindata[j][i] - mean[trainlabel[j]][i])**2
   
    for i in range(10):
        for j in range(784):
            variance[i][j] /= numlabel[i]
            if variance[i][j] < 10:
                variance[i][j] = 10

    return variance



def showpic(img):
    temp = [[0 for i in range(28)] for _ in range(28)]
    for i in range(28):
        for j in range(28):
            temp[i][j] = img[(i*28)+j]
    plt.imshow(temp, cmap='gray')
    plt.savefig('num.jpg')



def draw(traindata, trainlabel):
    #calculate prior
    numlabel = calculate_numlabel(trainlabel)
    numbin = calculate_numbin(traindata, trainlabel)
    prior = calculate_prior(numlabel, numbin)

    f = open('img.txt', 'w')
    for i in range(10):
        for j in range(784):
            zero = 0
            one = 0
            for k in range(16):
                zero += prior[i][j][k]
            for k in range(16, 32):
                one += prior[i][j][k]
            
            
            if(one >= zero):
                f.write('1 ')
            else:
                f.write('0 ')
            
            if(((j+1) % 28) == 0):
                f.write('\n')
        f.write('\n\n')


def classifier(traindata, trainlabel, testdata, testlabel, mode):
    if mode == 0:
        #calculate prior
        numlabel = calculate_numlabel(trainlabel)
        numbin = calculate_numbin(traindata, trainlabel)
        prior = calculate_prior(numlabel, numbin)
        acc = 0
        f = open('output_mode0.txt', 'w')

        for x in range(len(testlabel)):
            #calculate probability
            probability = [0] * 10
            for i in range(10):
                for j in range(784):
                    probability[i] += math.log(prior[i][j][testdata[x][j]])

                probability[i] += math.log(numlabel[i] / len(trainlabel))
            total = sum(probability)
            
            #因負號被消掉，所以最後要選數字小的
            for i in range(10):
                probability[i] /= total


            f.write('Postiror (in log scale) :\n')
            for i in range(10):
                f.write(f'{i}: {probability[i]}\n')
            f.write(f'prediction: {probability.index(min(probability))}, ans: {testlabel[x]}\n')
            if probability.index(min(probability)) == testlabel[x]:
                acc += 1
            
            
        f.write(f'err rate : {(10000 -acc)/10000}\n')
        print(acc/10000)   
        f.close


    if mode == 1:
        f = open('output_mode1.txt', 'w')
        acc = 0
        #label1~9的數量
        numlabel = calculate_numlabel(trainlabel)
        print(f'numlabel = {numlabel}')
        #每個label的每個特徵的mean和mu
        mean = calculate_mean(traindata, trainlabel, numlabel)
        variance = calculate_variance(traindata, trainlabel, mean, numlabel)
        #P(1~9)
        for i in range(10):
            numlabel[i] = math.log(numlabel[i] / len(trainlabel))

        #P(1|1~784)、P(2|1~784)、、、、、
        #log(P(1|784)) = log(P(1)) -(1/2)sigma[log(2*Pi*vatiance^2)]-(1/2)sigma[((x-mu)^2)/varianve^2]
        for i in range(len(testdata)):
            probability = [0] * 10
            for j in range(10):
                for k in range(784):
                    probability[j] += math.log(2 * math.pi * variance[j][k])
                    probability[j] += ((testdata[i][k] - mean[j][k])**2) / variance[j][k]
                probability[j] *= (-1/2)
                probability[j] += numlabel[j]

            total = sum(probability)

            #因負號被消掉，所以最後要選數字小的
            for j in range(10):
                probability[j] /= total


            f.write('Postiror (in log scale) :\n')
            
            for j in range(10):

                f.write(f'{j}: {probability[j]}\n')
            #P(1|1~784)、 、、、、P(9|1~784) 選機率最大的

            f.write(f'prediction: {probability.index(min(probability))}, ans: {testlabel[i]}\n')
            if probability.index(min(probability)) == testlabel[i]:
                acc += 1
            
        f.write(f'err rate : {(10000 -acc)/10000}\n')
        print(acc/10000)   
        f.close
                    
        
        


#load data
trainImgFile = 'train-images.idx3-ubyte'
trainLabelFile = 'train-labels.idx1-ubyte'
testImgFile = 't10k-images.idx3-ubyte'
testLabelFile = 't10k-labels.idx1-ubyte'
trainImg = read_img(trainImgFile)
trainLabel = read_label(trainLabelFile)
testImg = read_img(testImgFile)
testLabel = read_label(testLabelFile)
showpic(trainImg[0])
print(f'label[0] = {trainLabel[0]}')
print(f'labellist size = {len(trainLabel)}, imglist size = {len(trainImg)}')


classifier(trainImg, trainLabel, testImg, testLabel, 1)
classifier(trainImg, trainLabel, testImg, testLabel, 0)
draw(trainImg, trainLabel)

