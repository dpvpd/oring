# -*- coding: utf-8 -*-
# 파이썬에서 # 문자 뒤에 오는 것은 주석입니다. C언어 및 C계열 언어에서의 //와 같은 역할을 합니다.
# 라이브러리를 불러옵니다. os, sys, pickle, math는 파이썬 내장 라이브러리입니다.
# 내장 라이브러리가 아닌 라이브러리는 주석으로 달아둔 명령어를 터미널 또는 CMD(명령프롬프트)에 입력함으로써 설치할 수 있습니다.
from PyQt5 import QtCore, QtGui, QtWidgets, uic  #python -m pip install PyQt5
import os
import sys
import cv2 as cv  #python -m pip install opencv-python, python -m pip install numpy
import pickle
import math

dir = os.getcwd() + '/' #프로그램의 경로를 문자열로 저장
 
# PyQt5 라이브러리의 UI 렌더링을 위해 main.ui파일을 불러옴
# main.ui 파일은 https://github.com/dpvpd/oring 에 업로드됨
# .ui파일은 QtDesigner 프로그램을 이용해 작성했으며, xml언어를 잘 알지 못해 설명하지 않음. 
# QtDesigner 프로그램으로 .ui 파일을 열어보면 쉽게 그 구조를 파악할 수 있음. 
# PyQt5 라이브러리를 설치할 때 QtDesigner 또한 함께 설치됨. 
ui = uic.loadUiType(dir+'main.ui')[0]
 
_translate = QtCore.QCoreApplication.translate  # pyqt5에 텍스트를 띄울 때 필요한 함수
distance = lambda p1,p2:(((p1[0]-p2[0])**2)+((p1[1]-p2[1])**2))**.5  #두 점의 평면좌표가 주어졌을 때 직선 거리를 구하는 람다식
area = lambda p1,p2:abs(p1[0]-p2[0])*abs(p1[1]-p2[1])  #직사각형의 마주보는 두 점의 좌표를 입력하면 직사각형의 넓이를 구하는 람다식
ftypes = 'image files (*.png *.jpeg *.jpg *.bmp);; all files (*.*)'  #불러올 사진의 확장자, 추가하고싶으면 ... *.bmp *.tga 와 같이 (공백)*.(확장자) 로 입력하면 됨 
fontSize,textOrg = 2,(10,80)  #이미지를 출력할 때 좌측 상단에 쓰여질 텍스트의 크기와 좌표(해당 점을 좌상단 꼭짓점으로 텍스트를 그림)
 
 
class Form(QtWidgets.QMainWindow, ui):  #GUI 루프를 갖는 클래스, 앞서 불러온 main.ui 파일을 인수로 가짐
    def __init__(self):  # 클래스가 호출될 때 실행되는 함수
        super().__init__()  # 부모 클래스인 QMainWindow 루프의 init함수를 이 함수로 대체
        self.setFixedSize(1213,584)  # UI 창의 사이즈를 해당 크기로 고정
        self.setupUi(self)  # UI를 화면에 표시하기위해 렌더링함
        self.loadOptionValues() # 저장해둔 변수 값을 불러와 UI의 각 위치에 값을 넣어주는 함수. 아래쪽에 정의되어있음.
        self.output = ''  # 출력 문자열 저장 변수
        self.pushButton.clicked.connect(self.takeAPicture) # "Shot"이라고 쓰여진 버튼을 눌렀을 때 Form 클래스의 takeAPicture 함수를 실행함.
        self.pushButton_2.clicked.connect(self.saveOptionValues) # "Save"라고 쓰여진 버튼을 눌렀을 때 UI의 각 요소의 값을 파일로 저장하는 함수를 실행함.
        self.FindFile.clicked.connect(self.setFilePath) # "..."이라고 쓰여진 버튼을 눌렀을 때 이미지를 선택하는 창을 띄우는 함수를 실행함.
        self.pushButton_3.clicked.connect(self.loadImage) # "Load"라고 쓰여진 버튼을 눌렀을 때 Form 클래스의 loadImage 함수를 실행함.
        self.pushButton_4.clicked.connect(exit) # "Exit"라고 쓰여진 버튼을 눌렀을 때 프로그램을 종료함. 
        self.show() # 위와 같은 내용으로 렌더링된 UI 루프를 사용자에게 보여줌. 즉, 프로그램 창을 실행시킴.

    # 이미지 파일의 경로를 지정하는 함수. "..."라고 쓰여진 버튼을 눌렀을 때 호출됨
    def setFilePath(self):
        path = QtWidgets.QFileDialog.getOpenFileName(self, 'Select Image File',dir,ftypes)[0].replace('\\','/')
        # self : 부모 윈도우를 의미함. 'Select Image File' : 선택 창의 타이틀(창 위에 뜨는 텍스트)임. dir : 창이 처음 열렸을 때 보여줄 디렉토리
        # ftypes = 'image files (*.png *.jpeg *.jpg *.bmp);; all files (*.*)', 파일 확장자를 지시함.
        self.FilePath.setText(_translate("Dialog", path)) # 위 창에서 선택된 파일의 경로를 "..." 버튼 좌측의 텍스트 위젯에 입력함. 
        return

    # 창 좌측 여러 변수들의 값을 불러오는 함수. 실행시 최초 1회만 작동함. 
    def loadOptionValues(self):
        try: # try-except문은 C언어의 TRY{} CATCH{}문과 같은 역할임. 
            #아래 줄에서 'optionsave.pyvars'라는 이름의 파일을 열려고 시도할 때, 해당 파일이 존재하지 않으면 except 문 내용으로 넘어감. 
            with open('optionsave.pyvars','rb') as f:
                self.blurStdDev.setValue(pickle.load(f))
                self.houghResolution.setValue(pickle.load(f))
                self.MINRAD.setValue(pickle.load(f))
                self.param1.setValue(pickle.load(f))
                self.param2.setValue(pickle.load(f))
                self.houghMinrad.setValue(pickle.load(f))
                self.houghMaxrad.setValue(pickle.load(f))
                self.circleRange.setValue(pickle.load(f))
                self.distStdDev.setValue(pickle.load(f))
                self.blurStdDev_2.setValue(pickle.load(f))
                self.distance.setValue(pickle.load(f))
                self.CamAngle.setValue(pickle.load(f))
                self.FilePath.setText(pickle.load(f))
                # 각각의 숫자 및 문자열 입력 위젯에 불러온 값을 대입함. (setValue : 숫자, setText : 문자열)
        except:
            pass

    #창 좌측 여러 변수들의 값을 저장하는 함수. optionsave.pyvars 라는 파일 이름으로 저장됨. 
    def saveOptionValues(self):
        with open('optionsave.pyvars','wb') as f:
            pickle.dump(self.blurStdDev.value(),f)
            pickle.dump(self.houghResolution.value(),f)
            pickle.dump(self.MINRAD.value(),f)
            pickle.dump(self.param1.value(),f)
            pickle.dump(self.param2.value(),f)
            pickle.dump(self.houghMinrad.value(),f)
            pickle.dump(self.houghMaxrad.value(),f)
            pickle.dump(self.circleRange.value(),f)
            pickle.dump(self.distStdDev.value(),f)
            pickle.dump(self.blurStdDev_2.value(),f)
            pickle.dump(self.distance.value(),f)
            pickle.dump(self.CamAngle.value(),f)
            pickle.dump(self.FilePath.text(),f)
            # 각각의 숫자 및 문자열 입력 위젯에서 불러온 값을 저장함. 순서는 앞선 loadOptionValues 함수와 같음.
            
    # 입력된 contour가 원의 형태를 갖는지 판별함. 
    def isCircle(self,contours,centers) -> tuple:
        output = ''
        l = []
        for i in contours:
            l.append(distance(centers,list(i[0])))
        # 함수의 인수로 받은 contours리스트의 모든 원소에 대해 그 좌표와 중심점의 좌표 사이의 거리를 l 리스트에 저장함. 
        mean = sum(l)/len(l) # l 리스트 원소의 평균값임. 거리의 평균은 원의 반지름에 근사함
        a = 0
        for i in l:
            a+=(i-mean)**2
        sigma = (a/len(l))**.5 #표준편차를 구함. 
        output += 'Std.Dev. = '+str(sigma)+'\n'

        if sigma > self.distStdDev.value():
            return False,output
        return True,output
        #원이 맞으면 ture, 아니면 false와 함께 output 문자열을 리턴함. 

    def takeAPicture(self):
        cap = cv.VideoCapture(0) # 시스템의 1번째 카메라를 사용함
        while cap.isOpened(): 
            ret, img = cap.read()
            self.calc(img) #카메라에서 불러온 이미지를 calc 함수에 인수로 주어 calc함수 실행. 
            break
        cap.release() # 카메라 사용 종료
        

    def loadImage(self):
        FileName = self.FilePath.text()
        try:
            img = cv.imread(FileName) # "..."버튼 옆의 텍스트 상자에 입력된 문자열의 경로의 사진을 불러옴.
        except:
            QtWidgets.QMessageBox.critical(self,'Wrong File!','Wrotten File Path is Wrong. \nPlease check the Path or File is exist. ',QtWidgets.QMessageBox.Yes)
            # 만약 경로가 잘못되어 이미지를 불러올 수 없다면 경고창을 띄우고 함수를 종료함. 
            return

        self.calc(img) # 불러온 이미지를 calc함수에 인수로 주어 calc함수 실행.

    def calc(self,img):

        #각 이미지별 디스플레이에 표시하기 위해 리사이즈될 이미지 크기
        #labelsize = (781,561)  # 원본
        #labelsize = (781,439)  # 16:9 이미지
        labelsize = (748,561)  # 4:3 이미지
        #labelsize = (747,561)  # 라즈베리파이 V2 카메라
        self.output = ''

        
        # 인수로 받은 이미지에 가우시안 블러 필터를 적용함. 
        # img : 인수로 받은 원본 이미지임
        # (0,0) : 커널의 크기. 반드시 (홀수,홀수)꼴로 주어져야하며 (0,0)의 경우 시그마값에 따라 정해짐
        # self.blurStdDev.value() : "Gaussian Std. Devi."라고 쓰여진 스핀박스(숫자 입력)의 값을 불러옴. 가우시안 블러의 시그마값임.
        # cv.GaussianBlur 함수에 대한 더 자세한 설명은 https://opencv-python.readthedocs.io/en/latest/doc/11.imageSmoothing/imageSmoothing.html 의 Gaussian Filtering 문항에서 확인할 수 있음. 
        img_blur = cv.GaussianBlur(img,(0,0),self.blurStdDev.value())

        # cvtColor 함수를 통해 앞서 가우시안 블러를 적용한 BGR 이미지를 Grayscale로 바꿈. 
        # cv.COLOR_BGR2GRAY를 이용해 연산 방식을 지정함.
        # 이후 BGR을 RGB로 바꾸어 출력할 때에도 쓰임. 
        img_gray = cv.cvtColor(img_blur,cv.COLOR_BGR2GRAY)

        # "Threshold"라고 쓰여진 선택창에서 선택된 임계값연산 방식에 따라 img_binary 변수를 만든다. 
        # binary 변수는 임계값연산 방식별 값을 저장하는 변수이다. 원래 opencv에 있는 cv.ADAPTIVE_THRESHOLD_GAUSSIAN_C 등과 같은 상수값들을 eval을 통해 저장해준다. 
        binary = eval(self.threshold.currentText())
        if 'ADAPTIVE' in self.threshold.currentText(): # ADAPTIVE_THRESH_GAUSSIAN_C 및 ADAPTIVE_THRESH_MEAN_C를 선택했을 때 adaptiveThreshold를 이용해 적응형임계값연산.
            img_binary = cv.adaptiveThreshold(img_blur, 255,binary,cv.THRESH_BINARY,15,2)
        else: # 적응형 두가지가 아니면 threshold를 이용해 임계값연산.
            ret, img_binary = cv.threshold(img_gray,127,255,binary)

        # 허프 서클 알고리즘으로 원으로 추정되는 물체의 좌표를 구한다. 
        # 본 함수의 인수에 대한 설명은 추가 문서로 서술함. 
        circles = cv.HoughCircles(img_binary,cv.HOUGH_GRADIENT,self.houghResolution.value(),self.MINRAD.value(),param1=self.param1.value(),param2=self.param2.value(),minRadius=self.houghMinrad.value(),maxRadius=self.houghMaxrad.value())

        #허프서클로 원이 검출되었을 때
        if circles is not None:
            howmanycircles = []
            #과다검출
            #if len(circles[0]) > 2:
            #    img = cv.putText(img,'Too many circles detected.',textOrg,cv.FONT_HERSHEY_SIMPLEX,fontSize,(0,255,0),4)
            #원이 3개 이상 검출되면 이미지상 오류로 판별하려 했으나 해당 기능을 삭제함.
            # 해당 기능을 복구하고자 한다면, # 기호를 빼 주석처리를 취소하고 아래의 if False: pass 를 삭제하면 된다. 
            if False:
                pass
            #1 or 2개 검출
            else:
                #사각형 범위 구하기
                rects = []
                for i in circles[0,:]:
                    aas = i[2] + self.circleRange.value()
                    r1 = int(i[0]-aas),int(i[1]-aas)
                    r2 = int(i[0]+aas),int(i[1]+aas)
                    rects.append((r1,r2))
                    
                #앞서 구한 사각형 범위에서 가장 큰 사각형을 rect변수에 저장
                rect_areas = []
                for i in range(len(rects)):
                    rect_areas.append((area(rects[i][0],rects[i][1]),rects[i]))
                rect_areas.sort(key=lambda x:x[0])
                rect = rect_areas[-1][1]

                #위에서 고른 사각형 범위만큼 크롭
                im = img[rect[0][1]:rect[1][1],rect[0][0]:rect[1][0]]
                
                #크롭한 이미지 블러, 흑백, 이진화 (앞서 img 변수의 이미지에 행한 연산 그대로, 과정중 입력받는 변수 값은 threshold 방식 선택 아래쪽에 따로. )
                try:
                    im_blur = cv.GaussianBlur(im,(1,1),self.blurStdDev_2.value())
                except:
                    im_blur = cv.medianBlur(im,int(self.blurStdDev_2.value()))
                im_gray = cv.cvtColor(im_blur,cv.COLOR_BGR2GRAY)
                ret, im_binary = cv.threshold(im_gray,127,255,binary)

                # 크롭한 이미지 컨투어 검출
                # im_binary : 컨투어를 검출할 이미지. cv.RETR.LIST : 검색방법. cv.CHAIN_APPROX_NONE : 근사화 방법
                # 추가 검색 방법 및 근사화 방법은 https://076923.github.io/posts/Python-opencv-21/#추가-정보 에서 볼 수 있음. 
                contours, hierarchy = cv.findContours(im_binary,cv.RETR_LIST,cv.CHAIN_APPROX_NONE)
                centers = []

                # 모든 컨투어에 대해 연산 시행
                for c in range(len(contours[:])):
                    cv.drawContours(im,[contours[c]],0,(0,255,0),1) # 이미지에 윤곽선을 그림.
                    cnt = contours[c]

                    cnx,cny,cnw,cnh = cv.boundingRect(cnt) # 컨투어에 외접하는 직사각형의 좌상단 좌표 및 너비, 높이
                    cnx += rect[0][0] # img에서 크롭한 im에서의 좌표이므로, img에서의 im의 위치를 고려해 img에서의 위치를 계산
                    cny += rect[0][1]
                    img = cv.rectangle(img,(cnx,cny),(cnx+cnw,cny+cnh),(0xC0,0xFF,0xEE),2) # 컨투어에 외접하는 직사각형을 그림. im이 아닌 img에 그림. 

                    # 카메라 화각과 거리를 통해 물체의 가로 크기를 계산하는 부분임. 단순 수학 계산만 수행함. 
                    dis = self.distance.value()
                    ang = self.CamAngle.value()/2
                    ih, iw, ichannel = img.shape
                    x1,x2 = cnx,cnx+cnw
                    r = 0
                    if x1==x2 or iw<2:
                        r = 0
                    else:
                        center = iw//2
                        r1 = ((x1/center)-1)*ang
                        r2 = ((x2/center)-1)*ang
                        if r1==0:
                            r = dis*math.tan(math.radians(r2))
                        elif r2==0:
                            r = dis*math.tan(math.radians(abs(r1)))
                        elif r1<0 and r2<0:
                            r = dis*(math.tan(math.radians(abs(r1)))-math.tan(math.radians(abs(r2))))
                        elif r1>0 and r2>0:
                            r = dis*(math.tan(math.radians(abs(r2)))-math.tan(math.radians(abs(r1))))
                        elif r1<0 and r2>0:
                            r = dis*(math.tan(math.radians(abs(r2)))+math.tan(math.radians(abs(r1))))
                        else:
                            r = 0

                    # 컨투어 중심 좌표 구하기
                    M = cv.moments(contours[c])
                    try:
                        cx = int(M['m10']/M['m00'])
                    except ZeroDivisionError:
                        cx = int(M['m10'])
                    try:
                        cy = int(M['m01']/M['m00'])
                    except ZeroDivisionError:
                        cy = int(M['m01'])
                    centre = (cx,cy)
                    centers.append(centre)

                    # 원 판별, 원이면 반지름과 표준편차 저장
                    asdfasdf, asdfasdfasdf = self.isCircle(contours[c],centre)
                    print(r)
                    if r>0:
                        if asdfasdf:
                            img = cv.putText(img,'width %f'%(r),(cnx,cny),cv.FONT_HERSHEY_SIMPLEX,fontSize/2,(255,0,0),4)
                            self.output += 'Diameter = '+str(r)+'\n'
                            self.output += asdfasdfasdf
                            self.output+='\n'
                            howmanycircles.append(True)
                        else:
                            howmanycircles.append(False)

                    cv.circle(im,(cx,cy),2,(255,0,0),-1)

                # 이미지에 원이 검출되었는지 출력하기
                self.ValueOutput.setText(_translate("MainWindow", self.output))
                if False not in howmanycircles: # 모든 검출된 컨투어가 원일 때
                    img = cv.putText(img,'Perfect Circle.',textOrg,cv.FONT_HERSHEY_SIMPLEX,fontSize,(255,0,0),4)
                elif True not in howmanycircles: # 모든 검출된 컨투어가 원이 아니거나 컨투어가 검출되지 않았을 때
                    img = cv.putText(img,'Not a Circle.',textOrg,cv.FONT_HERSHEY_SIMPLEX,fontSize,(0,0,255),4)
                else: # 검출된 컨투어들 중 일부만 원일 때
                    img = cv.putText(img,'%d circles were detected.'%(howmanycircles.count(True)),textOrg,cv.FONT_HERSHEY_SIMPLEX,fontSize,(0,255,0),4)

                #이미지에 사각형 그리기
                for i in rects:
                    cv.rectangle(img,i[0],i[1],(127,255,0),3)

        #이미지 출력
        img1 = cv.resize(img, dsize=labelsize, interpolation=cv.INTER_AREA)
        img1 = cv.cvtColor(img1,cv.COLOR_BGR2RGB) # opencv는 이미지의 색상 정보를 BGR로 불러오는데, PyQt5는 RGB 이미지를 출력하므로 BGR을 RGB로 바꿈. 
        h,w,c = img1.shape
        qImg = QtGui.QImage(img1.data, w, h, w*c, QtGui.QImage.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(qImg)
        self.imglabel.setPixmap(pixmap)
 
      
 
if __name__ == '__main__':
   app=QtWidgets.QApplication(sys.argv)
   form = Form()
   sys.exit(app.exec_())
