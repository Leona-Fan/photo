# coding: utf-8
import numpy
from PyQt5.QtCore import QObject, pyqtSignal


class SignalBus(QObject):
    errorSignal = pyqtSignal(str)
    updateMainCameraSignal = pyqtSignal(object)
    testSignal = pyqtSignal()
    quitSignal = pyqtSignal()
    gestureSignal = pyqtSignal(str)


signalBus = SignalBus()
