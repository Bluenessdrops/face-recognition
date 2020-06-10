import cv2, sys
from PyQt5.QtWidgets import *
from PyQt5.QtCore import QTimer, pyqtSignal, QObject
from PyQt5.QtGui import QImage, QPixmap, QMovie, QTextCursor
from CNN_train import cnnTrain
from CNN_recg  import displays
from func_file import coordinate, pre_process, points, predictor

num = 0

class Emtstrm(QObject):
    textWritten = pyqtSignal(str)  #定义一个发送str的信号
    def write(self, text):
        self.textWritten.emit(str(text))#text即显示的文本


class Example(QWidget):

    def __init__(self):
        super().__init__()
        self.initUI()
        self.initTimer()

    def for_button(self):
        self.LBL.setEnabled(True)
        self.vc = cv2.VideoCapture(0)
        self.vc.set(3,320)
        self.vc.set(4,240)
        self.OCBTN1.setEnabled(False)
        self.OCBTN2.setEnabled(False)
        self.OCBTN3.setEnabled(False)
        self.OCBTN4.setEnabled(False)
        self.CCBTN.setEnabled(True)

    def cap_mine(self):
        """捕捉我的脸"""
        self.for_button()
        self.face_path = 'E:/FRTM/Face/Face of me'
        self.timer.start(1)

    def cap_others(self):
        """捕捉别人的脸"""
        self.for_button()
        self.face_path = 'E:/FRTM/Face/Face of others/from camera'
        self.timer.start(1)

    def cnn_recognition(self):
        self.for_button()
        self.face_path = 0
        self.timer.start(1)

    def cnn_train(self):
        cnnTrain()

    def closeCamera(self):
        """关闭摄像头"""
        global num
        num = 0

        self.vc.release()
        self.OCBTN1.setEnabled(True)
        self.OCBTN2.setEnabled(True)
        self.OCBTN3.setEnabled(True)
        self.OCBTN4.setEnabled(True)
        self.CCBTN.setEnabled(False)
        self.QLable_close()
        self.timer.stop()

    def initTimer(self):
        """定时器"""
        self.timer = QTimer(self)#初始化
        self.timer.timeout.connect(self.show_pic)#计时结束调用show_pic展示图片

    def show_pic(self):
        """opencv捕获图片"""
        global num

        num+=1
        ret, img = self.vc.read()
        img = cv2.flip(img, 1)#翻转
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)#单通道灰度图
        a, b, c, d, dlibed = coordinate(gray)#翻转的

        if  a != 0 and d != 0:#face captured
            if self.face_path != 0 :#cap mine
                pre_process(a, b, c, d, gray, num, self.face_path)#翻转的单通道灰度图
                img = points(img, dlibed, predictor)
            elif self.face_path == 0:#cnn recg
                img = displays(a,b,c,d,img, gray)

        cur_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)#翻转的，三通道彩色图
        height, width = cur_frame.shape[:2]
        pixmap = QImage(cur_frame, width, height, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(pixmap)
        self.LBL.setPixmap(pixmap)

    def outputWritten(self, text):
        cursor = self.TXTED.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.insertText(text)
        self.TXTED.setTextCursor(cursor)
        self.TXTED.ensureCursorVisible()

    def QLable_close(self):
        self.gif = QMovie('./GUI/back_pic/background.gif')
        self.LBL.setMinimumSize(640,480)
        self.LBL.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.LBL.setPixmap(QPixmap())
        self.LBL.setMovie(self.gif)
        self.gif.start()

    def initUI(self):
        """定义窗体"""
        sys.stdout = Emtstrm(textWritten=self.outputWritten)
        sys.stderr = Emtstrm(textWritten=self.outputWritten)

        self.OCBTN4 = QPushButton('CNN recg')
        self.OCBTN4.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        self.OCBTN4.clicked.connect(self.cnn_recognition)#undefined

        self.OCBTN3 = QPushButton('CNN train')
        self.OCBTN3.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        self.OCBTN3.clicked.connect(self.cnn_train)#undefined

        self.OCBTN2 = QPushButton('cap others')
        self.OCBTN2.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        self.OCBTN2.clicked.connect(self.cap_others)#undefined

        #self.OCBTN1.setStyleSheet("QPushButton{border-image: url(./GUI/back_pic/1.png)}")
        self.OCBTN1 = QPushButton('cap mine')
        self.OCBTN1.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        self.OCBTN1.clicked.connect(self.cap_mine)#undefined

        self.CCBTN = QPushButton('terminate')
        self.CCBTN.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        self.CCBTN.clicked.connect(self.closeCamera)

        self.OCBTN1.setEnabled(True)
        self.OCBTN2.setEnabled(True)
        self.OCBTN3.setEnabled(True)
        self.OCBTN4.setEnabled(True)
        self.CCBTN .setEnabled(False)

        self.LBL = QLabel()
        self.LBL.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.LBL.setScaledContents (True)  # 让图片自适应label大小

        self.TXTED = QTextEdit()
        self.TXTED.setReadOnly(True)
        self.TXTED.setMinimumSize(640,40)
        self.TXTED.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.TXTED.setStyleSheet("background-color: #eee5db;")

        self.hbox  = QHBoxLayout(self)# define layout
        self.vboxx = QVBoxLayout(self)
        self.vboxx.addWidget(self.LBL)# add wid
        self.vboxx.addWidget(self.TXTED)

        self.vbox = QVBoxLayout(self)#define layout

        self.vbox.addWidget(self.OCBTN1)#add wids
        self.vbox.addWidget(self.OCBTN2)
        self.vbox.addWidget(self.OCBTN3)
        self.vbox.addWidget(self.OCBTN4)
        self.vbox.addWidget(self.CCBTN)

        self.hbox.addLayout(self.vboxx)#add layout
        self.hbox.addLayout(self.vbox)#add layout

        self.setLayout(self.hbox)
        self.QLable_close()
        self.setWindowTitle('face recognition program')
        self.setFixedSize(740,600)
        self.setGeometry(300, 300, 700, 500)
        self.show()


if __name__=='__main__':
    app = QApplication(sys.argv)
    ex = Example()
    sys.exit(app.exec_())
