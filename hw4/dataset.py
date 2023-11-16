import numpy as np

imgdata = 'train-images.idx3-ubyte'
labeldata = 'train-labels.idx1-ubyte'

def read(imgdata, labeldata):
    with open(imgdata, 'rb') as f:
        magic = f.read(4)
        imgnum = int.from_bytes(f.read(4), byteorder='big')
        rownum = int.from_bytes(f.read(4), byteorder='big')
        colnum = int.from_bytes(f.read(4), byteorder='big')
        data = []

        for i in range(imgnum):
            img = np.zeros((rownum, colnum))
            for r in range(rownum):
                for c in range(colnum):
                    img[r, c] = int.from_bytes(f.read(1), byteorder='big')
            data.append(img)

        data = np.array(data)

    with open(labeldata, 'rb') as f:
        magic = f.read(4)   
        labelnum = int.from_bytes(f.read(4), byteorder='big')
        
        label = np.array([int.from_bytes(f.read(1)) for _ in range(labelnum)])

    np.save(f'{imgdata}.npy', data)
    np.save(f'{labeldata}.npy', label)

if __name__ == '__main__':
    read(imgdata, labeldata)