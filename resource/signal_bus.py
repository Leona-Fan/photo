# coding: utf-8
import numpy
from PyQt5.QtCore import QObject, pyqtSignal
from PyQt5.QtWidgets import QLabel


class SignalBus(QObject):
    errorSignal = pyqtSignal(str)
    updateMainCameraSignal = pyqtSignal(object)
    testSignal = pyqtSignal()
    quitSignal = pyqtSignal()
    gestureSignal = pyqtSignal(str, QLabel)


signalBus = SignalBus()
