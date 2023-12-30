# _*_ coding: utf-8 _*_
# @Author   : Wei Yue
# @Time     : 2023-12-30 8:30
# @Function :
# _*_ coding: utf-8 _*_
# @Author   : Wei Yue
# @Time     : 2023-12-21 12:58
# @Function :  UI界面
import os
import random
import sys

import numpy as np

import HandTrackingModule as htm
import cv2
from PyQt5.QtCore import Qt, QTimer, pyqtSlot
from PyQt5.QtGui import QIcon, QImage, QPixmap, QKeySequence
from PyQt5.QtWidgets import QApplication, QShortcut
from qfluentwidgets import InfoBar, InfoBarPosition, MSFluentTitleBar
from qfluentwidgets.components.widgets.frameless_window import FramelessWindow
import pickle

from photo.common.constant import STYLE
from resource.signal_bus import signalBus
from ui.Ui_camMain import Ui_camMain
import copy
import itertools
import mediapipe as mp


def exception_hook(type, value, tb):
    error_title = f"Exception Type: {type}\nException Value: {value}\n"
    error_message = 'Traceback Information:\n'
    while tb:
        frame = tb.tb_frame
        error_message += f"  File: {frame.f_code.co_filename}, line {tb.tb_lineno}, in {frame.f_code.co_name}\n"
        tb = tb.tb_next
    signalBus.errorSignal.emit(error_title)


class XianyuFaceDetc(FramelessWindow, Ui_camMain):

    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.setTitleBar(MSFluentTitleBar(self))
        self.setWindowTitle('Face Gesture Detector')
        self.setWindowIcon(QIcon(':/icon/FaceDetect.svg'))
        pixmap = QPixmap(':/icon/Photo.svg')
        self.styleLbl.setPixmap(pixmap)
        self.lastPhotoLbl.setPixmap(pixmap)
        self.pic_dir = f'./pics/{random.choice(STYLE)}'
        self.pic_file_list = [os.path.join(self.pic_dir, file_name) for file_name in os.listdir(self.pic_dir)]
        pixmap = QPixmap(random.choice(self.pic_file_list)).scaled(200,200)
        self.styleLbl.setPixmap(pixmap)
        # init dir
        self.init_dir()
        # init resource
        self.init_resource()

        # init short-cut
        self.quitShortcut = QShortcut(QKeySequence("Q"), self)
        # init signal button:
        self.connectSignalToSlot()

    def init_dir(self):
        DATA_DIR_list = ['./imgs', './photos']
        for dir in DATA_DIR_list:
            if not os.path.exists(dir):
                os.makedirs(dir)

    def init_resource(self):
        self.cap = cv2.VideoCapture(0)
        self.timer = QTimer(self)
        # 画面刷新周期为 30ms
        self.timer.start(30)
        # sampleCount = 5代表150ms 识别一次
        self.sample_count = 5
        self.sample = 0
        # 加载Haar级联分类器
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        # pTime = 0
        self.detector = htm.handDetector(detectionCon=0.7)
        # 读入训练好的model.p文件
        model_dict = pickle.load(open('./model.p', 'rb'))
        self.model = model_dict['model']
        # 分类标签
        self.labels_dict = {0: '0', 1: '1', 2: '2', 3: '3'}
        self.hands = mp.solutions.hands.Hands(static_image_mode=True, min_detection_confidence=0.7)
        self.photo_count = 0

    def connectSignalToSlot(self):
        signalBus.errorSignal.connect(self.createTopInfoBar)
        signalBus.quitSignal.connect(self.quit)
        signalBus.testSignal.connect(self.test)
        signalBus.gestureSignal.connect(self.gestureProcess)
        self.timer.timeout.connect(self.updateMainCamera)
        self.quitShortcut.activated.connect(self.quit)

    @pyqtSlot()
    def test(self):
        print(1111)

    @pyqtSlot()
    def quit(self):
        self.timer.stop()
        self.releaseResoure()
        pixmap = QPixmap(':/icon/Camera.svg')
        self.cameraLbl.setPixmap(pixmap)

    @pyqtSlot(str)
    def gestureProcess(self, gesture):
        print(gesture)

    @pyqtSlot()
    def updateMainCamera(self):
        success, frame = self.cap.read()
        cv2.putText(frame, "Pose", (440, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
        if self.sample == self.sample_count:
            self.sample = 0
            H, W, _ = frame.shape
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(frame_rgb)
            # 将图像转为灰度
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # 检测人脸
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
            x_ = []
            y_ = []
            pre_processed_landmark_list = None
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y
                        x_.append(x)
                        y_.append(y)
                    landmark_list = self.calc_landmark_list(frame, hand_landmarks)
                    # 归一化
                    pre_processed_landmark_list = self.pre_process_landmark(
                        landmark_list)
                    # 检测手的关键点，draw=False
                    frame = self.detector.findHands(frame, draw=False)
                prediction = self.model.predict([np.array(pre_processed_landmark_list)])
                predicted_character = self.labels_dict[int(prediction[0])]
                # 相邻两张照片时间间隔至少为0.3秒
                if predicted_character in ['0', '1', '2'] and len(faces) > 0:
                    signalBus.gestureSignal.emit(predicted_character)
                    for (x, y, w, h) in faces:
                        cv2.imwrite(f'photos/photo_{self.photo_count}.jpg', frame)
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                        self.photo_count += 1
                        cv2.putText(frame, f"Photo Count: {self.photo_count}, pose: {predicted_character}", (10, 70),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        # 显示到主界面
        self.sample += 1
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = image.shape
        bytes_per_line = ch * w
        q_image = QImage(image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        self.cameraLbl.setPixmap(pixmap)

    # 骨架坐标映射为像素点
    def calc_landmark_list(self, image, landmarks):
        image_width, image_height = image.shape[1], image.shape[0]

        landmark_point = []

        for _, landmark in enumerate(landmarks.landmark):
            landmark_x = min(int(landmark.x * image_width), image_width - 1)
            landmark_y = min(int(landmark.y * image_height), image_height - 1)

            landmark_point.append([landmark_x, landmark_y])

        return landmark_point

    # 骨架坐标归一化
    def pre_process_landmark(self, landmark_list):
        temp_landmark_list = copy.deepcopy(landmark_list)

        base_x, base_y = 0, 0
        for index, landmark_point in enumerate(temp_landmark_list):
            if index == 0:
                base_x, base_y = landmark_point[0], landmark_point[1]

            temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
            temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

        temp_landmark_list = list(
            itertools.chain.from_iterable(temp_landmark_list))

        # 归一化
        max_value = max(list(map(abs, temp_landmark_list)))

        def normalize_(n):
            return n / max_value

        temp_landmark_list = list(map(normalize_, temp_landmark_list))

        return temp_landmark_list

    def releaseResoure(self):
        self.cap.release()
        cv2.destroyAllWindows()

    def createTopInfoBar(self, msg: str):
        InfoBar.error(
            title=self.tr('ERROR!'),
            content=msg,
            orient=Qt.Horizontal,
            isClosable=True,
            position=InfoBarPosition.TOP,
            duration=5000,
            parent=self
        )


if __name__ == '__main__':
    QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)
    # sys.excepthook = exception_hook
    app = QApplication(sys.argv)
    w = XianyuFaceDetc()
    w.show()
    app.exec_()
