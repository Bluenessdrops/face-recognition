import cv2
from random import uniform, randint
from dlib import get_frontal_face_detector as GFFD
from dlib import shape_predictor

gffd = GFFD()
cut_pixel = 64                               #剪裁的像素大小
predictor = shape_predictor("F:/Program Files/Python3.7.6/Lib/site-packages/face_recognition_models/models/shape_predictor_68_face_landmarks.dat")

#检测人脸区域
def coordinate(sample):#翻转的单通道灰度图
    dlibed = gffd(sample, 1)                #调用dlib的识别函数得到人脸区域的rectangles
    if  not  dlibed:                        #当未检测到人脸时dlibed会返回 rectangles[]
        a = b = c = d = 0                   #当未检测到人脸时强行给出value来return，且要和下面主函数的对应
    else:
        for x, pot in enumerate(dlibed):
            a = pot.top() if pot.top() > 0 else 0
            b = pot.left() if pot.left() > 0 else 0
            c = pot.right() if pot.right() > 0 else 0
            d = pot.bottom() if pot.bottom() > 0 else 0

    return a, b, c, d, dlibed

#图像预处理函数
def pre_process(a, b, c, d, sample, num, face_path):#翻转的单通道灰度图
    gray = cv2.cvtColor(sample, cv2.COLOR_GRAY2RGB)#翻转的三通道灰度图
    cut_area  = gray[a:d, b:c]
    cut_ops   = cv2.resize(cut_area, (cut_pixel, cut_pixel))
    adjust    = re_lighted(cut_ops, uniform(0.9, 1.5), randint(-40, 20))
    cv2.imwrite(face_path+'/'+str(num)+'.jpg', adjust)
    print(f'{num} image have saved.')

#随机打光函数
def re_lighted(image, lightness, contrast):
    width  = image.shape[1]
    height = image.shape[0]
    for i in range(width):                              #依次遍历每一列像素
        for j in range(height):
            for k in range(3):
                temp = int(image[j, i, k]*lightness + contrast)
                if temp > 255:                          #避免报错
                    temp = 255
                elif temp < 0:
                    temp = 0
                image[j, i, k] = temp                   #得到处理之后的图像
    return image                                        #返回值返回到上面的pre_process函数中进行灰度处理

#显示从图像或视频或摄像头捕获的图片
def display(a, b, c, d, wdnm, img, num, face_path):#仅翻转
    if  a == 0 and d == 0:
        cv2.imshow(wdnm, img)
    else :
        flip = cv2.rectangle(img, (b, a), (c, d), (0, 255, 0), 1)
        cv2.imshow(wdnm, flip)

    if  a != 0 and d != 0:
        num += 1
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        pre_process(a, b, c, d, img, num, face_path)#翻转的单通道灰度图

    return num


def points(image, dlibed, predictor):
    for face in dlibed:
        shape = predictor(image, face)
    for pots in shape.parts():
        potx = (pots.x, pots.y)
        image = cv2.circle(image, potx, 1, (0,255,0),1)
    return image


