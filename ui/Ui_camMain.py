# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'camMain.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_camMain(object):
    def setupUi(self, camMain):
        camMain.setObjectName("camMain")
        camMain.resize(1000, 550)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/icon/FaceDetect.svg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        camMain.setWindowIcon(icon)
        self.horizontalLayout_7 = QtWidgets.QHBoxLayout(camMain)
        self.horizontalLayout_7.setObjectName("horizontalLayout_7")
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setSizeConstraint(QtWidgets.QLayout.SetDefaultConstraint)
        self.gridLayout.setContentsMargins(20, 30, 20, 20)
        self.gridLayout.setSpacing(12)
        self.gridLayout.setObjectName("gridLayout")
        self.frame = QtWidgets.QFrame(camMain)
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout(self.frame)
        self.horizontalLayout_4.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.CardWidget_4 = CardWidget(self.frame)
        self.CardWidget_4.setObjectName("CardWidget_4")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.CardWidget_4)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.cameraLbl = QtWidgets.QLabel(self.CardWidget_4)
        self.cameraLbl.setStyleSheet("border-radius: 10px; background-color: lightblue; padding: 10px;")
        self.cameraLbl.setText("")
        self.cameraLbl.setScaledContents(False)
        self.cameraLbl.setAlignment(QtCore.Qt.AlignCenter)
        self.cameraLbl.setObjectName("cameraLbl")
        self.horizontalLayout.addWidget(self.cameraLbl)
        self.horizontalLayout_4.addWidget(self.CardWidget_4)
        self.verticalLayout_3 = QtWidgets.QVBoxLayout()
        self.verticalLayout_3.setSpacing(0)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.referencePhotoCard = CardWidget(self.frame)
        self.referencePhotoCard.setObjectName("referencePhotoCard")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.referencePhotoCard)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.styleLbl = QtWidgets.QLabel(self.referencePhotoCard)
        self.styleLbl.setStyleSheet("border-radius: 10px; background-color: lightblue; padding: 10px;")
        self.styleLbl.setText("")
        self.styleLbl.setAlignment(QtCore.Qt.AlignCenter)
        self.styleLbl.setObjectName("styleLbl")
        self.horizontalLayout_2.addWidget(self.styleLbl)
        self.verticalLayout_3.addWidget(self.referencePhotoCard)
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_3.addItem(spacerItem)
        self.IconWidget_2 = IconWidget(self.frame)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.IconWidget_2.sizePolicy().hasHeightForWidth())
        self.IconWidget_2.setSizePolicy(sizePolicy)
        self.IconWidget_2.setMinimumSize(QtCore.QSize(15, 15))
        self.IconWidget_2.setMaximumSize(QtCore.QSize(20, 20))
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap(":/icon/Style.svg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.IconWidget_2.setIcon(icon1)
        self.IconWidget_2.setObjectName("IconWidget_2")
        self.horizontalLayout_3.addWidget(self.IconWidget_2)
        self.styleLblTitle = StrongBodyLabel(self.frame)
        self.styleLblTitle.setObjectName("styleLblTitle")
        self.horizontalLayout_3.addWidget(self.styleLblTitle)
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_3.addItem(spacerItem1)
        self.verticalLayout_3.addLayout(self.horizontalLayout_3)
        self.lastPhotoCard = CardWidget(self.frame)
        self.lastPhotoCard.setObjectName("lastPhotoCard")
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout(self.lastPhotoCard)
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.lastPhotoLbl = QtWidgets.QLabel(self.lastPhotoCard)
        self.lastPhotoLbl.setStyleSheet("border-radius: 10px; background-color: lightblue; padding: 10px;")
        self.lastPhotoLbl.setText("")
        self.lastPhotoLbl.setAlignment(QtCore.Qt.AlignCenter)
        self.lastPhotoLbl.setObjectName("lastPhotoLbl")
        self.horizontalLayout_5.addWidget(self.lastPhotoLbl)
        self.verticalLayout_3.addWidget(self.lastPhotoCard)
        self.verticalLayout_3.setStretch(0, 6)
        self.verticalLayout_3.setStretch(1, 1)
        self.verticalLayout_3.setStretch(2, 6)
        self.horizontalLayout_4.addLayout(self.verticalLayout_3)
        self.horizontalLayout_4.setStretch(0, 5)
        self.horizontalLayout_4.setStretch(1, 2)
        self.gridLayout.addWidget(self.frame, 0, 0, 1, 1)
        self.horizontalLayout_7.addLayout(self.gridLayout)

        self.retranslateUi(camMain)
        QtCore.QMetaObject.connectSlotsByName(camMain)

    def retranslateUi(self, camMain):
        _translate = QtCore.QCoreApplication.translate
        camMain.setWindowTitle(_translate("camMain", "camMain"))
        self.styleLblTitle.setText(_translate("camMain", "风格"))
from qfluentwidgets import CardWidget, IconWidget, StrongBodyLabel
import resource.resource_rc
