import struct as st
import numpy as np
import matplotlib.pyplot as plt
import math

# control parameter
is_test = False
is_conti = True
skip_pred = False
# < 0 to print all test case
print_case_num = 3
test_num = 200

# global variable
if not is_test:
    filename = {'images' : 'train-images.idx3-ubyte' ,\
                'labels' : 'train-labels.idx1-ubyte' ,\
                'test_img' : 't10k-images.idx3-ubyte' ,\
                'test_labels' : 't10k-labels.idx1-ubyte',\
                'online_test' : 'testfile.txt'}
else:
    filename = {'images' : 't10k-images.idx3-ubyte' ,\
                'labels' : 't10k-labels.idx1-ubyte',\
                'online_test' : 'testfile.txt'}

train_img_arr = []
train_label_arr = []
test_img_arr = []
test_label_arr = []
p_classes = []
var_arr = []
mean_arr = []
bin_prob_arr = []
classes = []
nImg = 0
nRow = 0
nCol = 0

# math tool
def C(N,m):
    return math.factorial(N)/(math.factorial(m) * math.factorial(N - m))

def gamma(n):
    if n == 1 or n == 2:
        return 1
    else:
        return math.factorial(n - 1)

def beta(a,b,p):
        return p**(a - 1) * (1 - p)**(b - 1) * gamma(a+b) / (gamma(a) * gamma(b))

# beta
class Beta:
    def __init__(self,a,b):
        self.a = a
        self.b = b
        self.p = -1

    def beta(self):
        return beta(self.a,self.b,self.p)

    def likelihood(self,N,m):
        # The probability of beservation
        self.p = m / N
        return C(N,m) * self.p**m * (1 - self.p)**(N - m)

    def prior(self):
        return

    def update(self,N,m):
        print("Beta prior:     a = {} b = {}".format(self.a,self.b))
        self.a += m
        self.b += (N - m)
        print("Beta posterior: a = {} b = {}\n".format(self.a,self.b))

def online_learning(a,b):
    online = Beta(a,b)
    # load test file
    with open(filename['online_test'],'r') as f:
        lines = f.readlines()
    for i,line in enumerate(lines):
        line = line.strip()
        N,m = len(line),line.count("1")
        print("case {}: {}".format(i + 1,line))
        print("Likelihood: {}".format(online.likelihood(N,m)))
        online.update(N,m)
        # if i == 1:
        #     break

# open MNIST
def open_img(filename):
    global nImg,nRow,nCol
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
    
    train_img_arr = open_img(filename['images'])
    train_label_arr = open_label(filename['labels'])
    

    if is_test:
        test_img_arr = train_img_arr[0:test_num]
        test_label_arr = train_label_arr[0:test_num]
    else:
        test_img_arr = open_img(filename['test_img'])
        test_label_arr = open_label(filename['test_labels'])

    print("Train img num : ",len(train_label_arr))
    print("Test img num : ",len(test_label_arr))

# show the image
def im_show(img,label,ax):
    ax.set_title('Label is {label}'.format(label=label))
    ax.imshow(img, cmap='gray')

# naive bayes
def dis_naive_bayes():
    global bin_prob_arr
    w,h = train_img_arr[0].shape
    bin_prob_arr = np.zeros((len(classes),w,h,32))
    print(bin_prob_arr.shape)
    # calc 32 bin probability of every label
    for i,c in enumerate(classes):
        c_idxes = np.where(train_label_arr == c)
        c_imgs = train_img_arr[c_idxes]
        hist , bin = np.apply_along_axis(lambda a: np.histogram(a, bins=range(0,257,8)), 0, c_imgs)
        for pix,h in np.ndenumerate(hist):
            # add min value avoid bin equal to 0
            h[h == 0] = 1
            t = h.sum()
            h = np.log(h / h.sum())
            # normalization
            # h = (h - np.min(h)) / np.ptp(h)
            bin_prob = bin_prob_arr[i]
            bin_prob[pix[0],pix[1],:] = h


def conti_naive_bayes():
    global mean_arr,var_arr
    # calc mean & varance of every label
    for c in classes:
        c_idxes = np.where(train_label_arr == c)
        c_imgs = train_img_arr[c_idxes]
        mean = np.mean(c_imgs,axis=0)
        var = np.var(c_imgs,axis=0)
        mean_arr.append(mean)
        var_arr.append(var)
    mean_arr = np.array(mean_arr)
    # smooth the var to eliminate the zero value
    var_arr = np.array(var_arr) + 1000

def naive_bayes(is_continuous):
    global p_classes,classes
    #Find the Unique Classes in the Data
    classes,counts=np.unique(train_label_arr,return_counts=True)
    #Number of Classes in the Data
    n_class=len(classes)
    print("MNIST info : ",nImg," ",nRow," ",nCol)
    print("Class num : ",n_class,"\n")
    # calc P(class)
    p_classes = counts / len(train_label_arr)
    print("P(Class) : ",p_classes)
    if is_continuous:
        conti_naive_bayes()
    else:
        dis_naive_bayes()

def predict(is_continuous):
    result = []
    c_prob = np.zeros(len(classes))
    if is_continuous:
        for i,img in enumerate(test_img_arr):
            # calc P(image | class)
            for j,c in enumerate(classes):
                var = var_arr[j]
                mean = mean_arr[j]
                ratio = 1 / np.sqrt(2 * np.pi * var) * np.exp(-np.square(img - mean)/(2 * var))
                ratio = np.sum(np.log(ratio))
                c_prob[j] = ratio
                # break
            idx = np.argmax(c_prob)
            result.append(classes[idx])
            # print result probability
            c_prob = c_prob / c_prob.sum()
            if i < print_case_num or print_case_num < 0:
                for j,p in enumerate(c_prob):
                    print("{}: {}".format(j,p))
                print("Prediction: {}, Ans: {}\n".format(result[i],test_label_arr[i]))
    else:
        for i,img in enumerate(test_img_arr):
            img = np.digitize(img, bins=range(0,257,8)) - 1
            # calc P(image | class)
            for j,c in enumerate(classes):
                for pix,color in np.ndenumerate(img):
                    bin_idx = color
                    c_prob[j] += bin_prob_arr[j,pix[0],pix[1],bin_idx]
            idx = np.argmax(c_prob)
            result.append(classes[idx])
            # print result probability
            c_prob = c_prob / c_prob.sum()
            if i < print_case_num or print_case_num < 0:
                for j,p in enumerate(c_prob):
                    print("{}: {}".format(j,p))
                print("Prediction: {}, Ans: {}\n".format(result[i],test_label_arr[i]))
    
    return np.array(result)

def error(pred,truth):
    return 1 - np.sum(pred == truth) / len(truth)

def draw_pred(is_continuous,class_idx,ax):
    img = np.zeros_like(test_img_arr[0])
    color_prob = np.zeros(256)
    cls = classes[class_idx]
    if is_continuous:
        var = var_arr[class_idx]
        mean = mean_arr[class_idx]
        for i,pix in np.ndenumerate(img):
            for color in range(0,256):
                prob = 1 / np.sqrt(2 * np.pi * var[i]) * np.exp(-np.square(color - mean[i])/(2 * var[i]))
                color_prob[color] = prob
            color_prob = color_prob / color_prob.sum()
            if np.argmax(color_prob) >= 128:
                img[i] = 1
            # accumulate the probability of 0~127 & 128~255 and then compare with them
            # p_black = color_prob[0:128].sum()
            # p_white = color_prob[128:].sum()
            # if p_white > p_black:
            #     img[i] = 1
    else:
        bin_prob = bin_prob_arr[class_idx]
        for i,pix in np.ndenumerate(img):
            for color in range(0,256):
                bin_idx = np.digitize(color, bins=range(0,257,8)) - 1
                prob = bin_prob[i[0],i[1],bin_idx]
                color_prob[color] = prob
                if np.argmax(color_prob) >= 128:
                    img[i] = 1
            # accumulate the probability of 0~127 & 128~255 and then compare with them
            # p_black = 0
            # p_white = 0
            # for color in range(0,128):
            #     bin_idx = np.digitize(color, bins=range(0,257,8)) - 1
            #     prob = bin_prob[i[0],i[1],bin_idx]
            #     p_black += prob
            # for color in range(128,256):
            #     bin_idx = np.digitize(color, bins=range(0,257,8)) - 1
            #     prob = bin_prob[i[0],i[1],bin_idx]
            #     p_white += prob
            # if p_white > p_black:
            #     img[i] = 1
    print(img)
    im_show(img,cls,ax)

# main
# Naive bayes
load()
naive_bayes(is_conti)
if not skip_pred:
    res = predict(is_conti)
    if is_test:
        print(res,test_label_arr)
    print("Error rate:",error(res,test_label_arr))
# draw imagnation of model
fig, axs = plt.subplots(2, 5)
for i in range(0,2):
    for j in range(0,5):
        num = i*5+j
        draw_pred(is_conti,num,axs[i,j])
# draw_pred(is_conti,4)
plt.show()

# Online learning
print("Input a of beta prior : ")
a = int(input())
print("Input b of beta prior : ")
b = int(input())
online_learning(a,b)
