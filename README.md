# deep learning based face recognition
使用tensorflow 2.1 GPU版本（1.5的应该也可以）搭建的卷积神经网络来做1:1的人脸识别。使用的库是亚洲人脸集CAS-PEAL-R1。

using tensorflow 2.1&GPU version to build CNN to achieve face recognition. but actually 1.x version of tf maight be also helpful cuz i disabled new feature of tf 2.x ver. 

亚洲人脸集CAS-PEAL-R1的下载地址(天翼云)：https://cloud.189.cn/t/qIzUner63aIj (访问码:nac5)

download url. maybe not useful if u don't use this app↑

使用了Dlib检测人脸，opencv和PyQT5做可视化效果，卷积神经网络做人脸比对。关于Dlib和nvidia cuda/cudnn的正确安装请参考这篇文章→https://blog.csdn.net/bluenessdrops/article/details/105025266

Dlib is used for face detection,and PyQT5 for GUI,CNN for recognition. as for correct install for Dlib/cuda/cudnn, plz read this article(i feel really sry if u can't read Chinese. my engish is not good enough for writing such an long blog):https://blog.csdn.net/bluenessdrops/article/details/105025266

根目录下的3个文件cap_frm_xx是用来从摄像头/图片/视频中抓取人脸用的。

the 3 "cap_frm_xx" files at root dir are writed for get faces from camera/image/video. well if u want to debug it i suggest u go for code2 folder.

根目录下的func_file是整个工程的公用代码块。调用其中的函数使用。

func_file at root dir is public code for whole program, which is used by importing functions inside. don't debug it.

根目录下的func_CNN是CNN的代码，由CNN_train和CNN_recg调用。

func_CNN at root dir is code block for building CNN, which is used by CNN_train & CNN_recg. don't debug it.


根目录下的CNN_train用来做模型训练，CNN_recg用来做比对识别。

CNN_train at root dir is for model training and CNN_recg is for recognition.don't debug it.

根目录下的UI_MAIN是整合了摄像头捕获图像、模型训练和人脸比对的集大成者。另外我在上传的时候把GUI用来做背景的gif不小心删了。。。

UI_MAIN intergrated cap_frm_cam,CNN_train and CNN_recg. if u don't want any GUI,just go for code2 folder, which has the single files to debug.

code2文件夹里面的都是独立调试文件。除了带func的两个文件，其他都可以独立运行。

虽然代码中已经给出了解决方案，但是调试的时候还是请注意修改文件保存路径。

watch for save path, change them before debug.such as the path for image,model file,temp file or sth

如果有问题需要up解决请在上述csdn的文章下留言。github经常不看

if u have question need to be solved, plz leave a message at my csdn(i just give u one of my article above). i rarely check my github.
