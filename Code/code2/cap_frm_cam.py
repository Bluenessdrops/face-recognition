import os, sys,cv2
from func_file import coordinate, points, predictor

#初始化所需参数
def begin():
    choz = input('my face?\n\n')
    if  choz == 'yes':
        face_path = 'E:/FRTM/Face/Face of me'
    else:
        face_path = 'E:/FRTM/Face/Face of others/Face from camera'
    if  not os.path.exists(face_path):           #若不存在则创建路径
        os.makedirs(face_path)

    cap  = cv2.VideoCapture(0)                   #调用内置摄像头
    cap.set(3, 320)                              #width
    cap.set(4, 240)                              #height
    wdnm = 'camera'                              #window name 的缩写
    cv2.namedWindow(wdnm, cv2.WINDOW_KEEPRATIO)  #AUTOSIZE
    cv2.resizeWindow(wdnm, 640, 480)
    cv2.moveWindow(wdnm,100,100)

    return wdnm, cap, face_path

#主函数z
def main():
    num = cal = sums = 0                                 #赋初值。indx是抽样间隔判定器，abcd为dlib识别出的人脸区域坐标
    while True:
        num+=1
        ret, image = cap.read()             #变量cap获取到的图像赋给image
        image = cv2.flip(image, 1)
        print(ret)
        a, b, c, d, dlibed, cal = coordinate(image, cal)      #从coordinate函数中获取到人脸区域的像素坐标abcd
        #num = display(a, b, c, d, wdnm, image, num, face_path)
        image = points(a, d, image, dlibed, predictor)
        cv2.imshow(wdnm,image)
        percent = (num-cal)/num
        sums = sums + percent
        if cv2.waitKey(1) == 113:
            sums = sums/num
            print('总捕获照片数为：'+str(num))
            print('平均检测率为：'+ str(sums))
            sys.exit(0)


wdnm, cap, face_path = begin()              #执行一遍begin函数得到不需要后续修改的初始值
main()
