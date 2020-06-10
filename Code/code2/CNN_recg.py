import os,sys, cv2,time
import tensorflow
import func_CNN
from func_file import coordinate

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
###################### CNN function finished #########################
######################################################################

def is_my_face(image):
    #print([image/255.0])
    res = sess.run(predict, feed_dict={x: [image/255.0], keep_prob_5:1.0, keep_prob_75: 1.0})
    if res[0] == 1:
        return True
    else:
        return False

def displays(a,b,c,d,img, right, wrong):#翻转的
    to_gray = cv2.cvtColor(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2RGB)#三通道灰度图
    face = cv2.resize(to_gray[a:d,b:c], (size,size))#翻转的
    ismyface = is_my_face(face)#三通道灰度图

    if ismyface == True:
        text = 'wei peng'
        color = (0,255,0)
        right += 1
    elif ismyface == False:
        text = 'unknown'
        color = (0,255,255)
        wrong += 1

    flip = cv2.rectangle(img, (b,a),(c,d), color, 1)
    flip = cv2.putText(flip, text, (b,d+10), cv2.FONT_HERSHEY_COMPLEX_SMALL,0.5, color, 1)
    return flip, right, wrong


def main():

    start = time.clock()
    cap  = cv2.VideoCapture(0)                   #调用内置摄像头
    cap.set(3, 320)                              #width
    cap.set(4, 240)                              #height
    wdnm = 'camera'                              #window name 的缩写
    cv2.namedWindow(wdnm, cv2.WINDOW_KEEPRATIO)  #AUTOSIZE
    cv2.resizeWindow(wdnm, 640, 480)
    cv2.moveWindow(wdnm,300,300)
    right = wrong = sums= percent = 0

    while True:
        ret, img = cap.read()
        img = cv2.flip(img, 1)
        a,b,c,d, dlibed, cal = coordinate(img, cal = 1)

        if a != 0 or d != 0:
            img, right, wrong = displays(a,b,c,d,img, right, wrong)
            percent = right/(right + wrong)
            sums = sums+percent
            if (right + wrong)%10 == 0:
                once = time.clock()
                print('正确率'+str(percent)+'\n'+'运行时间：'+str(once)+'\n')
        cv2.imshow(wdnm,img)

        if cv2.waitKey(1) == 113:
            end = time.clock()
            aver = sums/(right+wrong)
            print(f'程序运行了{(end-start)}秒')
            print(f'正确率平均值为{aver}')
            sys.exit(0)


main()
sess.close()