import cv2, os, random
import numpy as np
import tensorflow
from sklearn.model_selection import train_test_split

tf = tensorflow.compat.v1
tf.disable_v2_behavior()


def getPaddingSize(img):
    h, w, _ = img.shape
    top, bottom, left, right = (0,0,0,0)
    longest = max(h, w)

    if w < longest:
        tmp = longest - w
        # //表示整除符号
        left = tmp // 2
        right = tmp - left
    elif h < longest:
        tmp = longest - h
        top = tmp // 2
        bottom = tmp - top
    else:
        pass
    return top, bottom, left, right

def readData(size, imgs, labs, path):
    for (path, dirnames, filenames) in os.walk(path):
        for filename in filenames:
            if filename.endswith('.jpg'):
                filename = path+'/'+filename
                img = cv2.imread(filename)

                top,bottom,left,right = getPaddingSize(img)
                # 将图片放大， 扩充图片边缘部分
                img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0,0,0])
                img = cv2.resize(img, (size, size))#height,width

                imgs.append(img)
                labs.append(path)

def whatthe(size, imgs, labs, my_faces_path, other_faces_path):
    readData(size, imgs, labs, my_faces_path)
    readData(size, imgs, labs,other_faces_path)
    # 将图片数据与标签转换成数组
    imgs = np.array(imgs)
    labs = np.array([[0,1] if lab == my_faces_path else [1,0] for lab in labs])
    # 随机划分测试集与训练集
    train_x,test_x,train_y,test_y = train_test_split(imgs, labs, test_size=0.05, random_state=random.randint(0,100))
    # 参数：图片数据的总数，图片的高、宽、通道
    train_x = train_x.reshape(train_x.shape[0], size, size, 3)
    test_x = test_x.reshape(test_x.shape[0], size, size, 3)
    # 将数据转换成小于1的数
    train_x = train_x.astype('float32')/255.0
    test_x = test_x.astype('float32')/255.0

    print('train size:%s, test size:%s' % (len(train_x), len(test_x)))
    # 图片块，每次取100张图片
    batch_size = 128####
    num_batch = len(train_x) // batch_size

    x = tf.placeholder(tf.float32, [None, size, size, 3])
    y_ = tf.placeholder(tf.float32, [None, 2])

    keep_prob_5 = tf.placeholder(tf.float32)
    keep_prob_75 = tf.placeholder(tf.float32)
    '''
    print('num_batch'+str(num_batch), '\n\nkeep_prob_5'+str(keep_prob_5),
    '\n\nkeep_prob_75'+str(keep_prob_75), '\n\ny_'+str(y_), '\n\ntrain_x'+str(train_x), '\n\ntrain_y'+str(train_y),
    '\n\nbatch_size'+str(batch_size), '\n\nx'+str(x), '\n\ntest_x'+str(test_x), '\n\ntest_y'+str(test_y))
    '''
    return num_batch, keep_prob_5, keep_prob_75, y_, train_x, train_y, batch_size, x, test_x, test_y

def weightVariable(shape):
    init = tf.random_normal(shape, stddev=0.01)
    return tf.Variable(init)

def biasVariable(shape):
    init = tf.random_normal(shape)
    return tf.Variable(init)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def maxPool(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

def dropout(x, keep):
    return tf.nn.dropout(x, keep)

def cnnLayer(x, keep_prob_5, keep_prob_75):
    # 第一层
    W1 = weightVariable([3,3,3,32]) # 卷积核大小(3,3)， 输入通道(3)， 输出通道(32)
    b1 = biasVariable([32])
    # 卷积
    conv1 = tf.nn.relu(conv2d(x, W1) + b1)
    # 池化
    pool1 = maxPool(conv1)
    # 减少过拟合，随机让某些权重不更新
    drop1 = dropout(pool1, keep_prob_5)

    # 第二层
    W2 = weightVariable([3,3,32,64])
    b2 = biasVariable([64])
    conv2 = tf.nn.relu(conv2d(drop1, W2) + b2)
    pool2 = maxPool(conv2)
    drop2 = dropout(pool2, keep_prob_5)

    # 第三层
    W3 = weightVariable([3,3,64,64])
    b3 = biasVariable([64])
    conv3 = tf.nn.relu(conv2d(drop2, W3) + b3)
    pool3 = maxPool(conv3)
    drop3 = dropout(pool3, keep_prob_5)

    # 全连接层
    Wf = weightVariable([8*8*64, 512])
    bf = biasVariable([512])
    drop3_flat = tf.reshape(drop3, [-1, 8*8*64])
    dense = tf.nn.relu(tf.matmul(drop3_flat, Wf) + bf)
    dropf = dropout(dense, keep_prob_75)

    # 输出层
    Wout = weightVariable([512,2])
    bout = biasVariable([2])
    #out = tf.matmul(dropf, Wout) + bout
    out = tf.add(tf.matmul(dropf, Wout), bout)
    return out
