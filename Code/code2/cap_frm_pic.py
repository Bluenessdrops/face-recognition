import sys, os, cv2#, keyboard
from func_file import coordinate, display

def begin():

    in_path = 'E:/FRTM/Sucai/image'
    out_path = 'E:/FRTM/Face/Face of others/Face from image'
    #in_path = 'E:/FRTM/Sucai/image'
    #out_path = 'E:/FRTM/Face/Face of others/Face from image'
    if  not os.path.exists(out_path):
        os.makedirs(out_path)

    wdnm = 'picture'
    cv2.namedWindow(wdnm, cv2.WINDOW_KEEPRATIO)  #AUTOSIZE
    cv2.resizeWindow(wdnm, 640, 480)
    cv2.moveWindow(wdnm,100,100)

    return wdnm, in_path, out_path


def main(num):

    for (path, dirnames, filenames) in os.walk(in_path):
        for filename in filenames:
            if filename.endswith('.tif'):
                img_path = path+'/'+filename
                img = cv2.imread(img_path)

                a, b, c, d = coordinate(img)
                num = display(a, b, c, d, wdnm, img, num, out_path)

                sys.exit(0) if cv2.waitKey(1) == 113 else None
               #sys.exit(0) if keyboard.is_pressed('q') else None


wdnm, in_path, out_path = begin()
main(num=0)