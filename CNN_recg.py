import os,sys, cv2
import tensorflow
import func_CNN

tf = tensorflow.compat.v1
tf.disable_v2_behavior()

size = 64

x = tf.placeholder(tf.float32, [None, size, size, 3])
keep_prob_5 = tf.placeholder(tf.float32)
keep_prob_75 = tf.placeholder(tf.float32)

output = func_CNN.cnnLayer(x, keep_prob_5, keep_prob_75)
predict = tf.argmax(output, 1)

saver = tf.train.Saver()
sess = tf.Session()
saver.restore(sess, tf.train.latest_checkpoint('E:/FRTM/CNN/CNN_model'))


######################################################################

def is_my_face(image):
    #print([image/255.0])
    res = sess.run(predict, feed_dict={x: [image/255.0], keep_prob_5:1.0, keep_prob_75: 1.0})
    if res[0] == 1:
        return True
    else:
        return False

def displays(a,b,c,d,img, gray):#翻转的
    to_gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)#三通道灰度图
    face = cv2.resize(to_gray[a:d,b:c], (size,size))#翻转的
    ismyface = is_my_face(face)#三通道灰度图
    if ismyface == True:
        text = 'wei peng'
        color = (0,255,0)
    elif ismyface == False:
        text = 'unknown'
        color = (0,255,255)

    flip = cv2.rectangle(img, (b,a),(c,d), color, 1)
    flip = cv2.putText(flip, text, (b,d+10), cv2.FONT_HERSHEY_COMPLEX_SMALL,0.5, color, 1)
    return flip

