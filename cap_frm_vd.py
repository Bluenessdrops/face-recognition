import os, sys, cv2#, keyboard
from func_file import coordinate, display


def begin():
    in_path ='E:/FRTM/Sucai/video/stn.3gp'
    cap  = cv2.VideoCapture(in_path)
    name = os.path.basename(in_path)
    face_path = 'E:/FRTM/Face/Face of others/Face from video/'+ name

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
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            a, b, c, d, dlibed = coordinate(gray)
            num = display(a, b, c, d, wdnm, image, num, face_path)

            sys.exit(0) if cv2.waitKey(1) == 113 else None
           #sys.exit(0) if keyboard.is_pressed('q') == 113 else None
        else:
            break


wdnm, cap, face_path = begin()
main(num=0)