import cv2, os, random
import numpy as np
import tensorflow
from sklearn.model_selection import train_test_split

tf = tensorflow.compat.v1
tf.disable_v2_behavior()


def getPaddingSize(img):
    '''
        为了得到图片边缘补齐的像素值，然后将图片转成长和宽一样大小的正方形
    '''
    h, w, _ = img.shape
    top, bottom, left, right = (0,0,0,0)
    longest = max(h, w)

    if w < longest:
        tmp = longest - w
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
    '''
        填充边界函数。填充值由getPaddingSize函数决定(0)
    '''
    for (path, dirnames, filenames) in os.walk(path):
        for filename in filenames:
            if filename.endswith('.jpg'):
                filename = path+'/'+filename
                img = cv2.imread(filename)

                top,bottom,left,right = getPaddingSize(img)
                    # 将图片放大, 扩充图片边缘部分
                img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0,0,0])
                img = cv2.resize(img, (size, size))#height,width

                imgs.append(img)
                labs.append(path)

def whatthe(size, imgs, labs, my_faces_path, other_faces_path):
    '''
        batch_size: 每次训练使用的图片数量\n
        num_batch: batch的数量\n
        keep_prob: dropout用到的droprate\n
        train_x,train_y: 训练集测试集样本\n
        test_x,test_y: 训练集测试集标签\n
        x,y_,keep_prob: 占位符
    '''
    readData(size, imgs, labs, my_faces_path)
    readData(size, imgs, labs, other_faces_path)
        # 将图片数据与标签转换成数组
    imgs = np.array(imgs)
    labs = np.array([[0,1] if lab == my_faces_path else [1,0] for lab in labs])
        # 随机划分训练集和测试集
    train_x,test_x,train_y,test_y = train_test_split(imgs, labs, test_size=0.05, random_state=random.randint(0,100))
        # 参数:图片数据的总数,图片的高、宽、通道
    train_x = train_x.reshape(train_x.shape[0], size, size, 3)
    test_x = test_x.reshape(test_x.shape[0], size, size, 3)
        # 将数据转换成小于1的数
    train_x = train_x.astype('float32')/255.0
    test_x = test_x.astype('float32')/255.0

    print('train size:%s, test size:%s' % (len(train_x), len(test_x)))
        # 图片块,每次取128张图片
    batch_size = 128
        # batch的总数
    num_batch = len(train_x) // batch_size
        # placeholder函数是在神经网络构建graph的时候在模型中的占位,
        # 此时并没有把要输入的数据传入模型,它只会分配必要的内存。等建立session,
        # 在会话中,运行模型的时候通过feed_dict()函数向占位符喂入数据。
    x = tf.placeholder(tf.float32, [None, size, size, 3])#none表示可以喂入任意形状的张量。
    y_ = tf.placeholder(tf.float32, [None, 2])

    keep_prob_5 = tf.placeholder(tf.float32)
    keep_prob_75 = tf.placeholder(tf.float32)

    return num_batch, keep_prob_5, keep_prob_75, y_, train_x, train_y, batch_size, x, test_x, test_y



def weightVariable(shape):
    '''
        \n设置权重参数，从正态分布中输出随机值.
        \ntf.Variable:初始化变量,生成tensor,可以作为graph中其他op的输入
        \nshape:输出张量的形状
        \nmean:正态分布的均值默认为0
        \nstddev:正态分布的标准差默认为1.0
        \ndtype:输出类型默认为tf.float32
        \nseed:随机数种子,设置之后每次生成的随机数都一样
        \nname:操作的名称
    '''
    init = tf.random_normal(shape, stddev=0.01)
    return tf.Variable(init)

def biasVariable(shape):
    '''
        \n设置偏置参数，从正态分布中输出随机值.
        \ntf.Variable:生成tensor
        \nshape: 输出张量的形状
        \nmean: 正态分布的均值
        \nstddev: 正态分布的标准差
        \ndtype: 输出的类型
        \nseed: 随机数种子
        \nname: 操作的名称
    '''
    init = tf.random_normal(shape)
    return tf.Variable(init)

def conv2d(x, W):
    '''
        \ntf.nn.conv2d (input, filter, strides, padding, use_cudnn_on_gpu=None, data_format=None, name=None)
        \nfilter:卷积核(需是张量),shape为[filter_height为卷积核高度, filter_weight为卷积核宽度, in_channel图像通道数且与input的一致, out_channels是卷积核数量 ]
        \nstrides:卷积时在图像每一维的步长(第一位和最后一位固定必须是1),[ 1, strides, strides, 1]
        \npadding:卷积的形式,“SAME”(考虑边界,不足的时候用0去填充周围)或“VALID”(不考虑)
        \nuse_cudnn_on_gpu:是否使用cudnn加速,默认为true
        \n返回tensor
    '''
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def maxPool(x):
    '''
        \ntf.nn.max_pool(value, ksize, strides, padding, name=None)
        \nvalue:池化的输入,一般池化层接在卷积层后面所以输入通常是feature map,[batch, height, width, channels]这样的shape
        \nksize:池化窗口的大小,取一个四维向量[1, height, width, 1],因为不在batch和channels上池化所以这两设为1
        \nstrides:类似卷积的步长,[1, stride,stride, 1]
        \npadding:类似卷积,取'VALID'或'SAME'
        \n返回一个Tensor,类型不变,shape仍然是[batch, height, width, channels]这种形式
    '''
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

def dropout(x, keep):
    '''
        \n一般用在全连接层.该函数就是使tensor中某些元素变为0,没变的元素变为1/keep_prob
        \ntf.nn.dropout(x, keep_prob, noise_shape=None, seed=None,name=None)
        \nx：指输入
        \nnoise_shape：一个1维的int32张量，代表了随机产生"保留/丢弃"标志的shape
        \nkeep_prob: 设置神经元被选中的概率,在初始化时keep_prob是一个占位符
    '''
    return tf.nn.dropout(x, keep)

def cnnLayer(x, keep_prob_5, keep_prob_75):
    # 第一层
    W1 = weightVariable([3,3,3,32]) # 卷积核大小(3,3), 输入通道(3), 输出通道(32)
    b1 = biasVariable([32])
    # 卷积
    conv1 = tf.nn.relu(conv2d(x, W1) + b1)#ReLU函数
    # 池化
    pool1 = maxPool(conv1)
    # 减少过拟合,随机让某些权重不更新
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

    '''
    # 第四层
    W4 = weightVariable([3,3,64,128])
    b4 = biasVariable([128])
    conv4 = tf.nn.relu(conv2d(drop3, W4) + b4)
    pool4 = maxPool(conv4)
    drop5 = dropout(pool4, keep_prob_5)
    '''

    # 全连接层
    Wf = weightVariable([8*8*64, 512])
    bf = biasVariable([512])
    drop3_flat = tf.reshape(drop3, [-1, 8*8*64])#-1表示函数会自动计算这个值
    dense = tf.nn.relu(tf.matmul(drop3_flat, Wf) + bf)
    dropf = dropout(dense, keep_prob_75)

    # 输出层
    Wout = weightVariable([512,2])
    bout = biasVariable([2])
    out = tf.add(tf.matmul(dropf, Wout), bout)#将矩阵a乘以矩阵b,生成a*b,并与c相加
    return out