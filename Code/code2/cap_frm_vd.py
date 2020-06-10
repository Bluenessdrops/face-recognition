import os, sys, cv2#, keyboard
from func_file import coordinate, display

#初始化所需参数
def begin():
    face_path = 'E:/FRTM/Face/Face of others/Face from video'

    cap = cv2.VideoCapture("E:/FRTM/Sucai/video/stn.mp4")

    wdnm = 'video'
    cv2.namedWindow(wdnm, cv2.WINDOW_KEEPRATIO)  #AUTOSIZE
    cv2.resizeWindow(wdnm, 640, 480)
    cv2.moveWindow(wdnm,100,100)

    if  not os.path.exists(face_path):
        os.makedirs(face_path)

    return wdnm, cap, face_path


def  main (num):
    while True :
        ret, image = cap.read()
        if ret == True:
            a, b, c, d = coordinate(image)
            num = display(a, b, c, d, wdnm, image, num, face_path)

            sys.exit(0) if cv2.waitKey(1) == 113 else None
        else:
            break


wdnm, cap, face_path = begin()
main(num=0)