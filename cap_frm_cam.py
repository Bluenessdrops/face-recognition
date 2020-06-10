import os, sys,cv2
from func_file import coordinate, display


def begin():
    choz = input('my face?\n\n')
    if  choz == 'yes':
        face_path = 'E:/FRTM/Face/Face of me/test3'
    else:
        face_path = 'E:/FRTM/Face/Face of others/Face from camera'
    if  not os.path.exists(face_path):
        os.makedirs(face_path)

    cap  = cv2.VideoCapture(0)
    cap.set(3, 320)
    cap.set(4, 240)
    wdnm = 'camera'
    cv2.namedWindow(wdnm, cv2.WINDOW_KEEPRATIO)  #AUTOSIZE
    cv2.resizeWindow(wdnm, 640, 480)
    cv2.moveWindow(wdnm,100,100)

    return wdnm, cap, face_path

#主函数
def main():
    num = 0
    while True:

        ret, image = cap.read()
        image = cv2.flip(image, 1)#翻转
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        a, b, c, d, dlibed = coordinate(gray)#翻转的单通道灰度图
        num = display(a, b, c, d, wdnm, image, num, face_path)#翻转的原图

        sys.exit(0) if cv2.waitKey(1) == 113 else None  #q == 113


wdnm, cap, face_path = begin()
main()
