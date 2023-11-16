import numpy as np

def dataprocess(img, imgnum, imgsize):
    t = np.zeros((imgnum, imgsize, imgsize))
    for n in range(imgnum):
        for i in range(imgsize):
            for j in range(imgsize):
                if img[n][i][j] > 127:
                    t[n][i][j] = 1
                else:
                    t[n][i][j] = 0
    return t

imgdata = 'train-images.idx3-ubyte.npy'
labeldata = 'train-labels.idx1-ubyte.npy'

img = np.load(imgdata)
label = np.load(labeldata)


imgnum = 60000
imgsize = 28

img = dataprocess(img, imgnum, imgsize)

lamb = np.full((10), 1/10)


p = np.zeros((10, imgsize, imgsize))
pre_p = np.zeros((10, imgsize, imgsize))

for n in range(10):
    for i in range(imgsize):
        for j in range(imgsize):
            p[n][i][j] = np.random.rand()/2 + 0.2



iteration = 0

while 1:
    iteration += 1
    w = np.zeros((imgnum, 10))

    #e step
    for n in range(imgnum):
        for i in range(10):
            w[n][i] = lamb[i] * np.prod(p[i] ** img[n]) * np.prod((1-p[i]) ** (1-img[n]))
        w[n] = w[n] / sum(w[n])

    #m step
    for n in range(10):

        lamb[n] = sum(w[:,n]) / imgnum

        wt = w[:,n].reshape(imgnum,1) 
        for i in range(imgsize):
            for j in range(imgsize):
                x = img[:, i, j].reshape(imgnum,1)
                p[n][i][j] = (wt.T @ x) / sum(w[:,n])  
                if p[n][i][j] == 0:
                    p[n][i][j] = 0.000001

    #draw img           
    f = open('output.txt','a')
    for n in range(10):
        f.write(f'class {n}:\n')
        for i in range(imgsize):
            for j in range(imgsize):
                if p[n][i][j] > 0.5:
                    f.write('1 ')
                else:
                    f.write('0 ')
            f.write('\n')
        f.write('\n')
    f.write(f'No. of iteration: {iteration}\n')
    f.write('-----------------------------------------\n')
    if(sum(sum(sum(abs(p-pre_p))))) < 15:
        break
    pre_p = p.copy()




#計算label[0~9]中class(0~9)各有幾個
count = np.zeros((10,10))
for i in range(imgnum):
    count[label[i]][np.argmax(w[i])] += 1


#挑出count中最大值的index，此值所在的i,j對應label[i] = class(j)
labeled = np.zeros((10))
for i in range(10):
    idx = np.argmax(count)
    class_idx = idx % 10
    actual_num = idx // 10
    print(f'actual_num = {actual_num}, class_idx = {class_idx}')
    labeled[int(actual_num)] = class_idx
    count[:,class_idx] = 0
    count[actual_num,:] = 0
    print(f'count = \n{count}')
print(f'labeled = \n{labeled}')

#draw img after classified
for n in range(10):
    f.write(f'labeled class {n}:\n')
    for i in range(imgsize):
        for j in range(imgsize):
            if p[int(labeled[n])][i][j] > 0.5:
                 f.write('1 ')
            else:
                f.write('0 ')
        f.write('\n')
    f.write('\n')



tp = np.zeros((10))
fp = np.zeros((10))
fn = np.zeros((10))
tn = np.zeros((10))

#confusion matrix
error = 0
for i in range(imgnum):
        
    actual = labeled[label[i]]
    predict = np.argmax(w[i])
    if actual == predict:
        tp[label[i]] += 1
        for j in range(10):
            if label[i] != j:
                tn[j] += 1
    else:
        error += 1
        fn[label[i]] += 1
        for j in range(10):
            if label[i] != j:
                fp[j] += 1
    
for i in range(10):
    f.write(f'confusion matrix {i}:\n')
    f.write(f'                  predict number {i} predict not number{i}\n')
    f.write(f'is number {i}            {tp[i]}             {fp[i]}\n')
    f.write(f'isn\'t number {i}        {fn[i]}             {tn[i]}\n')
    #sensitivity = TP /(TP+FN)
    #specificity = TN /(TN+FP)
    f.write(f'\nsensitivity (successfully predict number) {i} : {tp[i]/(tp[i]+fn[i])}\n')
    f.write(f'specificity (successfully predict not number) {i}: {tn[i]/(tn[i]+fp[i])}\n')
    f.write('--------------------------------------------\n')
f.write(f'total iteration to converge: {iteration}\n')
f.write(f'error rate: {error / imgnum}\n')
f.close()
