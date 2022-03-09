from cProfile import label
import struct as st
import numpy as np
import matplotlib.pyplot as plt

filename = {'images' : 'train-images.idx3-ubyte' ,\
            'labels' : 'train-labels.idx1-ubyte'}

img_arr = []
label_arr = []
# open MNIST
def load():
    global img_arr
    global label_arr
    train_imagesfile = open(filename['images'],'rb')
    train_imagesfile.seek(0)
    magic = st.unpack('>4B',train_imagesfile.read(4))
    nImg = st.unpack('>I',train_imagesfile.read(4))[0] #num of images
    nR = st.unpack('>I',train_imagesfile.read(4))[0] #num of rows
    nC = st.unpack('>I',train_imagesfile.read(4))[0] #num of column
    
    nBytesTotal = nImg*nR*nC*1 #since each pixel data is 1 byte
    print(nBytesTotal, " ", nImg," ",nR," ",nC)
    img_arr = np.asarray(st.unpack('>'+'B'*nBytesTotal,train_imagesfile.read(nBytesTotal))).reshape((nImg,nR,nC))

    with open('train-labels.idx1-ubyte', 'rb') as i:
        magic, size = st.unpack('>II', i.read(8))
        label_arr = np.fromfile(i, dtype=np.dtype(np.uint8)).newbyteorder(">")   

# show the image
def im_show(img,label):
    plt.title('Label is {label}'.format(label=label))
    plt.imshow(img, cmap='gray')
    plt.show()
# main

load()
im_show(img_arr[0],label_arr[0])