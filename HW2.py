from cProfile import label
import struct as st
from cv2 import sqrt
import numpy as np
import matplotlib.pyplot as plt
from sympy import factor

is_test = True
# global variable
if not is_test:
    filename = {'images' : 'train-images.idx3-ubyte' ,\
                'labels' : 'train-labels.idx1-ubyte' ,\
                'test_img' : 't10k-images.idx3-ubyte' ,\
                'test_labels' : 't10k-labels.idx1-ubyte'}
else:
    filename = {'images' : 't10k-images.idx3-ubyte' ,\
                'labels' : 't10k-labels.idx1-ubyte'}

train_img_arr = []
train_label_arr = []
test_img_arr = []
test_label_arr = []
p_classes = []
var_arr = []
mean_arr = []
classes = []
nImg = 0
nRow = 0
nCol = 0

# open MNIST
def open_img(filename):
    with open(filename,'rb') as imagesfile:
        imagesfile.seek(0)
        magic = st.unpack('>4B',imagesfile.read(4))
        nImg = st.unpack('>I',imagesfile.read(4))[0] #num of images
        nRow = st.unpack('>I',imagesfile.read(4))[0] #num of rows
        nCol = st.unpack('>I',imagesfile.read(4))[0] #num of column
        
        nBytesTotal = nImg*nRow*nCol*1 #since each pixel data is 1 byte
        img_arr = np.asarray(st.unpack('>'+'B'*nBytesTotal,imagesfile.read(nBytesTotal))).reshape((nImg,nRow,nCol))
        return img_arr

def open_label(filename):
    with open(filename, 'rb') as f:
        magic, size = st.unpack('>II', f.read(8))
        label_arr = np.fromfile(f, dtype=np.dtype(np.uint8)).newbyteorder(">")  
        return label_arr 

def load():
    global train_img_arr,train_label_arr
    global test_img_arr,test_label_arr
    global nImg,nRow,nCol
    
    train_img_arr = open_img(filename['images'])
    train_label_arr = open_label(filename['labels'])
    

    if is_test:
        test_img_arr = train_img_arr[0:10]
        test_label_arr = train_label_arr[0:10]
    else:
        test_img_arr = open_img(filename['test_img'])
        test_label_arr = open_label(filename['test_labels'])

# show the image
def im_show(img,label):
    plt.title('Label is {label}'.format(label=label))
    plt.imshow(img, cmap='gray')
    plt.show()

# naive bayes
def dis_naive_bayes():
    global p_classes,mean_arr,var_arr,classes

def conti_naive_bayes():
    global p_classes,mean_arr,var_arr,classes
    #Find the Unique Classes in the Data
    classes,counts=np.unique(train_label_arr,return_counts=True)
    #Number of Classes in the Data
    n_class=len(classes)
    print(nImg," ",nRow," ",nCol)
    print("Class num : ",n_class,"\n",counts)
    # calc P(class)
    p_classes = counts / len(train_label_arr)
    print(p_classes)
    # calc mean & varance of every label
    for c in classes:
        c_idxes = np.where(train_label_arr == c)
        c_imgs = train_img_arr[c_idxes]
        mean = np.mean(c_imgs,axis=0)
        var = np.var(c_imgs,axis=0)
        mean_arr.append(mean)
        var_arr.append(var)
        print(c,len(c_idxes[0]),c_imgs.shape,np.shape(var))
        print(c_imgs[:].shape)
    mean_arr = np.array(mean_arr)
    # smooth the var to eliminate the zero value
    var_arr = np.array(var_arr) + 1000

def naive_bayes(is_continuous):
    if is_continuous:
        conti_naive_bayes()
    else:
        dis_naive_bayes()

def predict():
    result = []
    for i,img in enumerate(test_img_arr):
        # calc P(image | class)
        c_prob = np.zeros(len(classes))
        for i,c in enumerate(classes):
            var = var_arr[i]
            mean = mean_arr[i]
            ratio = 1 / np.sqrt(2 * np.pi * var) * np.exp(-np.square(img - mean)/(2 * var))
            ratio = np.sum(np.log(ratio))
            c_prob[i] = ratio
            # break
        idx = np.argmax(c_prob)
        result.append(classes[idx])
    return np.array(result)

def error(pred,truth):
    return np.sum(pred == truth) / len(truth)

# wrong!!
def draw_pred(class_idx):
    cls = classes[class_idx]
    var = var_arr[class_idx]
    mean = mean_arr[class_idx]
    img = np.zeros_like(test_img_arr[0])
    for i,pix in np.ndenumerate(img):
        p_black = 0
        p_white = 0
        for color in range(0,128):
            prob = 1 / np.sqrt(2 * np.pi * var[i]) * np.exp(-np.square(color - mean[i])/(2 * var[i]))
            p_black += prob
        for color in range(128,256):
            prob = 1 / np.sqrt(2 * np.pi * var[i]) * np.exp(-np.square(color - mean[i])/(2 * var[i]))
            p_white += prob
        if p_white > p_black:
            img[i] = 1
    print(img)
    im_show(img,cls)

# main
load()
naive_bayes(True)
res = predict()
if is_test:
    print(res,test_label_arr)
print(error(res,test_label_arr))
draw_pred(1)
# im_show(train_img_arr[1],train_label_arr[0])