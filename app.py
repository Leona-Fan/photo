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
from PyQt5.QtWidgets import QApplication, QShortcut, QLabel
from qfluentwidgets import InfoBar, InfoBarPosition, MSFluentTitleBar
from qfluentwidgets.components.widgets.frameless_window import FramelessWindow
import pickle

from common.constant import STYLE, SAMPLE_COUNT, REFRESH_PERIOD, PHOTO_CONT_TIME, PHOTO_COUNT
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
        self.state = 1  # 1表示运行 0表示停止
        self.takingPhoto = 0
        pixmap = QPixmap(':/icon/Photo.svg')
        self.styleLbl.setPixmap(pixmap)
        self.lastPhotoLbl.setPixmap(pixmap)
        styleStr = random.choice(STYLE)
        self.pic_dir = f'./pics/{styleStr}'
        self.styleLblTitle.setText(styleStr)
        self.pic_file_list = [os.path.join(self.pic_dir, file_name) for file_name in os.listdir(self.pic_dir)]
        pixmap = QPixmap(random.choice(self.pic_file_list)).scaled(200, 200)
        self.styleLbl.setPixmap(pixmap)
        # init dir
        self.init_dir()
        # init resource
        self.init_resource()

        # init short-cut
        self.quitShortcut = QShortcut(QKeySequence("Q"), self)
        self.startShortcut = QShortcut(QKeySequence("S"), self)
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
        self.timer.start(REFRESH_PERIOD)
        self.timer_photo = QTimer(self)
        self.detect_sample = 0
        self.photo_sample = 0
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
        self.timer_photo.timeout.connect(self.updateTakingPhotoState)
        self.quitShortcut.activated.connect(self.quit)
        self.startShortcut.activated.connect(self.restart)

    @pyqtSlot()
    def test(self):
        print(1111)

    @pyqtSlot()
    def quit(self):
        if self.state == 0:
            return
        self.timer.stop()
        self.timer_photo.stop()
        self.releaseResoure()
        pixmap = QPixmap(':/icon/Camera.svg')
        self.cameraLbl.setPixmap(pixmap)
        self.state = 0

    @pyqtSlot()
    def restart(self):
        if self.state == 1:
            return
        self.init_resource()
        self.connectSignalToSlot()
        self.state = 1

    @pyqtSlot(str, QLabel)
    def gestureProcess(self, gesture, referPhoto: QLabel):
        if gesture == '1':
            styleStr = random.choice(STYLE)
            self.pic_dir = f'./pics/{styleStr}'
            self.styleLblTitle.setText(styleStr)
            self.pic_file_list = [os.path.join(self.pic_dir, file_name) for file_name in os.listdir(self.pic_dir)]
            pixmap = QPixmap(random.choice(self.pic_file_list)).scaled(200, 200)
            referPhoto.setPixmap(pixmap)
        elif gesture == '2':
            self.pic_file_list = [os.path.join(self.pic_dir, file_name) for file_name in os.listdir(self.pic_dir)]
            pixmap = QPixmap(random.choice(self.pic_file_list)).scaled(200, 200)
            referPhoto.setPixmap(pixmap)
        print('当前的手势是：' + gesture)

    @pyqtSlot()
    def updateTakingPhotoState(self):
        self.takingPhoto = 0
        self.timer_photo.stop()

    @pyqtSlot()
    def updateMainCamera(self):
        success, frame = self.cap.read()
        if not success:
            # 视频帧读取失败或帧为空，退出循环
            return
        cv2.putText(frame, "Pose", (440, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
        if self.detect_sample == SAMPLE_COUNT:
            self.detect_sample = 0
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
            predicted_character = ''
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
            if self.takingPhoto == 0 and len(predicted_character) > 0 and predicted_character != 0 and len(faces) > 0:
                print(predicted_character)
                signalBus.gestureSignal.emit(predicted_character, self.styleLbl)
            if predicted_character == '0' and len(faces) > 0:
                # 开启十秒拍照
                self.takingPhoto = 1
                self.timer_photo.start(PHOTO_CONT_TIME)

            if self.takingPhoto == 1 and len(faces) > 0 and self.photo_sample >= PHOTO_COUNT:
                self.photo_sample = 0
                for (x, y, w, h) in faces:
                    cv2.imwrite(f'photos/photo_{self.photo_count}.jpg', frame)
                    lastPhotoPixMap = QPixmap(f'photos/photo_{self.photo_count}.jpg').scaled(200, 200)
                    self.lastPhotoLbl.setPixmap(lastPhotoPixMap)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    self.photo_count += 1
                    cv2.putText(frame, f"Photo Count: {self.photo_count}, pose: {predicted_character}",
                                (10, 70),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        # 显示到主界面
        self.detect_sample += 1
        self.photo_sample += 1
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
