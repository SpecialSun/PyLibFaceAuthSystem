# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'ui_logs/admin_del.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.

import qtawesome
import sys
import img
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon


class AdminDeleteUi(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(538, 546)
        self.main_frame = QtWidgets.QFrame(Dialog)
        self.main_frame.setGeometry(QtCore.QRect(30, 20, 470, 470))
        self.main_frame.setMinimumSize(QtCore.QSize(470, 470))
        self.main_frame.setStyleSheet("\n"
                                      "#main_frame{\n"
                                      "background:#ffffff;\n"
                                      "border-top-left-radius:7px;\n"
                                      "border-top-right-radius:7px;\n"
                                      "border-bottom-left-radius:7px;\n"
                                      "border-bottom-right-radius:7px;\n"
                                      "border: 1px solid #8f8f91;\n"
                                      "\n"
                                      "}\n"
                                      "")
        self.main_frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.main_frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.main_frame.setObjectName("main_frame")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.main_frame)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setSpacing(0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.frame_2 = QtWidgets.QFrame(self.main_frame)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(self.frame_2.sizePolicy().hasHeightForWidth())
        self.frame_2.setSizePolicy(sizePolicy)
        self.frame_2.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_2.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_2.setObjectName("frame_2")
        self.close_button = QtWidgets.QPushButton(self.frame_2)
        self.close_button.setGeometry(QtCore.QRect(420, 10, 38, 38))
        self.close_button.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.close_button.setStyleSheet("QPushButton{\n"
                                        "                background-color: transparent; \n"
                                        "                border-radius:7px;\n"
                                        "  border: 2px solid #96989b;}\n"
                                        "            QPushButton:hover{\n"
                                        "               padding-top:5px;\n"
                                        "padding-left:5px;\n"
                                        "}")
        self.close_button.setText("")
        icon_close = qtawesome.icon('fa.times', color='#7b8290')
        self.close_button.setIcon(icon_close)
        self.close_button.setIconSize(QtCore.QSize(30, 30))  # 根据需要调整图标大小
        self.close_button.setObjectName("close_button")
        self.verticalLayout.addWidget(self.frame_2)
        self.frame_3 = QtWidgets.QFrame(self.main_frame)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(8)
        sizePolicy.setHeightForWidth(self.frame_3.sizePolicy().hasHeightForWidth())
        self.frame_3.setSizePolicy(sizePolicy)
        self.frame_3.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_3.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_3.setObjectName("frame_3")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.frame_3)
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_2.setSpacing(0)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.frame = QtWidgets.QFrame(self.frame_3)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(2)
        sizePolicy.setHeightForWidth(self.frame.sizePolicy().hasHeightForWidth())
        self.frame.setSizePolicy(sizePolicy)
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.frame)
        self.horizontalLayout.setContentsMargins(11, 0, -1, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.frame_5 = QtWidgets.QFrame(self.frame)
        self.frame_5.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_5.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_5.setObjectName("frame_5")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.frame_5)
        self.horizontalLayout_2.setContentsMargins(22, 0, 22, 0)
        self.horizontalLayout_2.setSpacing(0)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.frame_6 = QtWidgets.QFrame(self.frame_5)
        self.frame_6.setStyleSheet("")
        self.frame_6.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_6.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_6.setObjectName("frame_6")
        self.label_2 = QtWidgets.QLabel(self.frame_6)
        self.label_2.setGeometry(QtCore.QRect(190, 10, 111, 61))
        self.label_2.setStyleSheet("color:black;\n"
                                   "border:none;\n"
                                   "font-size:25px;\n"
                                   "font-weight:700;\n"
                                   "font-family: \"微软雅黑\", Helvetica, Arial, sans-serif;")
        self.label_2.setObjectName("label_2")
        self.label = QtWidgets.QLabel(self.frame_6)
        self.label.setGeometry(QtCore.QRect(120, 20, 51, 41))
        self.label.setText("")
        self.label.setPixmap(QtGui.QPixmap(":/icons/images/Logo.jpg"))
        self.label.setScaledContents(True)
        self.label.setObjectName("label")
        self.horizontalLayout_2.addWidget(self.frame_6)
        self.horizontalLayout.addWidget(self.frame_5)
        self.verticalLayout_2.addWidget(self.frame)
        self.frame_4 = QtWidgets.QFrame(self.frame_3)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(11)
        sizePolicy.setHeightForWidth(self.frame_4.sizePolicy().hasHeightForWidth())
        self.frame_4.setSizePolicy(sizePolicy)
        self.frame_4.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_4.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_4.setObjectName("frame_4")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.frame_4)
        self.verticalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_3.setSpacing(0)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.frame_7 = QtWidgets.QFrame(self.frame_4)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(8)
        sizePolicy.setHeightForWidth(self.frame_7.sizePolicy().hasHeightForWidth())
        self.frame_7.setSizePolicy(sizePolicy)
        self.frame_7.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_7.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_7.setObjectName("frame_7")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout(self.frame_7)
        self.horizontalLayout_4.setContentsMargins(-1, -1, 50, 0)
        self.horizontalLayout_4.setSpacing(0)
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.frame_9 = QtWidgets.QFrame(self.frame_7)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(4)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.frame_9.sizePolicy().hasHeightForWidth())
        self.frame_9.setSizePolicy(sizePolicy)
        self.frame_9.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_9.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_9.setObjectName("frame_9")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout(self.frame_9)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.register_msg_1 = QtWidgets.QPushButton(self.frame_9)
        self.register_msg_1.setEnabled(False)
        self.register_msg_1.setStyleSheet("color:black;\n"
                                          "text-align: right; \n"
                                          "border:none;\n"
                                          "font-size:22px;\n"
                                          "font-family: \"微软雅黑\";")
        self.register_msg_1.setObjectName("register_msg_1")
        self.verticalLayout_4.addWidget(self.register_msg_1)
        self.register_msg_2 = QtWidgets.QPushButton(self.frame_9)
        self.register_msg_2.setEnabled(False)
        self.register_msg_2.setStyleSheet("color:black;\n"
                                          "text-align: right; \n"
                                          "border:none;\n"
                                          "font-size:22px;\n"
                                          "font-family: \"微软雅黑\";")
        self.register_msg_2.setObjectName("register_msg_2")
        self.verticalLayout_4.addWidget(self.register_msg_2)
        self.horizontalLayout_4.addWidget(self.frame_9)
        self.frame_11 = QtWidgets.QFrame(self.frame_7)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(6)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.frame_11.sizePolicy().hasHeightForWidth())
        self.frame_11.setSizePolicy(sizePolicy)
        self.frame_11.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_11.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_11.setObjectName("frame_11")
        self.verticalLayout_5 = QtWidgets.QVBoxLayout(self.frame_11)
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.name_lineEdit = QtWidgets.QLineEdit(self.frame_11)
        self.name_lineEdit.setStyleSheet("border:2px solid gray;\n"
                                         "font-size:16px;\n"
                                         "font-weight:600;\n"
                                         "font-family: \"微软雅黑\";\n"
                                         "border-radius:12px;\n"
                                         "height:35px;")
        self.name_lineEdit.setInputMask("")
        self.name_lineEdit.setText("")
        self.name_lineEdit.setEchoMode(QtWidgets.QLineEdit.Normal)
        self.name_lineEdit.setObjectName("name_lineEdit")
        self.verticalLayout_5.addWidget(self.name_lineEdit)
        self.password_lineEdit = QtWidgets.QLineEdit(self.frame_11)
        self.password_lineEdit.setStyleSheet("border:2px solid gray;\n"
                                             "font-size:16px;\n"
                                             "font-weight:600;\n"
                                             "font-family: \"微软雅黑\";\n"
                                             "border-radius:12px;\n"
                                             "height:35px;")
        self.password_lineEdit.setText("")
        self.password_lineEdit.setEchoMode(QtWidgets.QLineEdit.Password)
        self.password_lineEdit.setObjectName("password_lineEdit")
        self.verticalLayout_5.addWidget(self.password_lineEdit)
        self.horizontalLayout_4.addWidget(self.frame_11)
        self.verticalLayout_3.addWidget(self.frame_7)
        self.frame_8 = QtWidgets.QFrame(self.frame_4)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(6)
        sizePolicy.setHeightForWidth(self.frame_8.sizePolicy().hasHeightForWidth())
        self.frame_8.setSizePolicy(sizePolicy)
        self.frame_8.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_8.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_8.setObjectName("frame_8")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(self.frame_8)
        self.horizontalLayout_3.setContentsMargins(90, 0, 90, 0)
        self.horizontalLayout_3.setSpacing(0)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.frame_10 = QtWidgets.QFrame(self.frame_8)
        self.frame_10.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_10.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_10.setObjectName("frame_10")
        self.verticalLayout_6 = QtWidgets.QVBoxLayout(self.frame_10)
        self.verticalLayout_6.setObjectName("verticalLayout_6")
        self.confirm_button = QtWidgets.QPushButton(self.frame_10)
        self.confirm_button.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.confirm_button.setStyleSheet("#confirm_button{\n"
                                          "color:#ffffff;\n"
                                          "background:#0099ff;\n"
                                          "border:2px solid gray;\n"
                                          "font-size:20px;\n"
                                          "\n"
                                          "font-family: \"微软雅黑\";\n"
                                          "border-radius:12px;\n"
                                          "height:38px;\n"
                                          "}\n"
                                          "\n"
                                          "#confirm_button:hover{\n"
                                          "padding-under:5px;\n"
                                          "padding-left:5px;\n"
                                          "background:#008deb;\n"
                                          "   }")
        self.confirm_button.setObjectName("confirm_button")
        self.verticalLayout_6.addWidget(self.confirm_button)
        self.cancel_button = QtWidgets.QPushButton(self.frame_10)
        self.cancel_button.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.cancel_button.setStyleSheet("#cancel_button{\n"
                                         "color:#ffffff;\n"
                                         "background:#0099ff;\n"
                                         "border:2px solid gray;\n"
                                         "font-size:20px;\n"
                                         "\n"
                                         "font-family: \"微软雅黑\";\n"
                                         "border-radius:12px;\n"
                                         "height:38px;\n"
                                         "}\n"
                                         "\n"
                                         "#cancel_button:hover{\n"
                                         "padding-under:5px;\n"
                                         "padding-left:5px;\n"
                                         "background:#008deb;\n"
                                         "   }")
        self.cancel_button.setObjectName("cancel_button")
        self.verticalLayout_6.addWidget(self.cancel_button)
        self.horizontalLayout_3.addWidget(self.frame_10)
        self.verticalLayout_3.addWidget(self.frame_8)
        self.verticalLayout_2.addWidget(self.frame_4)
        self.verticalLayout.addWidget(self.frame_3)
        lineText = [self.name_lineEdit, self.password_lineEdit]
        for lineEdit in lineText:
            lineEdit.setAlignment(Qt.AlignCenter)
        Dialog.setAttribute(QtCore.Qt.WA_TranslucentBackground)  # 隐藏外围边框
        Dialog.setWindowFlag(QtCore.Qt.FramelessWindowHint)  # 产生一个无边框的窗口，用户不能移动和改变大小
        Dialog.setWindowTitle("人脸识别")  # 设置标题
        Dialog.setWindowIcon(QIcon(':/icons/images/logo.jpg'))  # 设置logo
        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.label_2.setText(_translate("Dialog", "用户删除"))
        self.register_msg_1.setText(_translate("Dialog", "学号："))
        self.register_msg_2.setText(_translate("Dialog", "密码："))
        self.name_lineEdit.setPlaceholderText(_translate("Dialog", "请输入学号"))
        self.password_lineEdit.setPlaceholderText(_translate("Dialog", "请输入密码"))
        self.confirm_button.setText(_translate("Dialog", "删除"))
        self.cancel_button.setText(_translate("Dialog", "取消"))


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    widgets = QtWidgets.QMainWindow()
    ui = AdminDeleteUi()
    ui.setupUi(widgets)
    widgets.show()
    sys.exit(app.exec_())