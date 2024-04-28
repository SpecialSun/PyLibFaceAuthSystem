# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'zhuce.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.

import sys
import qtawesome
import img
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QIcon


class RegisterMsgUi(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(534, 714)
        self.main_frame = QtWidgets.QFrame(Dialog)
        self.main_frame.setGeometry(QtCore.QRect(30, 20, 470, 650))
        self.main_frame.setMinimumSize(QtCore.QSize(470, 650))
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
        self.close_button.setObjectName("close_button")
        icon_close = qtawesome.icon('fa.times', color='grey')
        self.close_button.setIcon(icon_close)
        self.close_button.setIconSize(QtCore.QSize(30, 30))  # 根据需要调整图标大小
        self.close_button.setObjectName("close_button")
        self.verticalLayout.addWidget(self.frame_2)
        self.frame_3 = QtWidgets.QFrame(self.main_frame)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(10)
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
        self.frame_6.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_6.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_6.setObjectName("frame_6")
        self.label_2 = QtWidgets.QLabel(self.frame_6)
        self.label_2.setGeometry(QtCore.QRect(100, 10, 271, 61))
        self.label_2.setStyleSheet("color:black;\n"
                                   "border:none;\n"
                                   "font-size:25px;\n"
                                   "font-weight:700;\n"
                                   "font-family: \"微软雅黑\", Helvetica, Arial, sans-serif;")
        self.label_2.setObjectName("label_2")
        self.label = QtWidgets.QLabel(self.frame_6)
        self.label.setGeometry(QtCore.QRect(40, 20, 51, 41))
        self.label.setText("")
        self.label.setPixmap(QtGui.QPixmap(":/icons/images/dog.png"))
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
        self.verticalLayout_4 = QtWidgets.QVBoxLayout(self.frame_7)
        self.verticalLayout_4.setContentsMargins(90, 0, 90, 0)
        self.verticalLayout_4.setSpacing(0)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.frame_9 = QtWidgets.QFrame(self.frame_7)
        self.frame_9.setStyleSheet("QLineEdit{\n"
                                   "border:2px solid gray;\n"
                                   "font-size:16px;\n"
                                   "font-weight:600;\n"
                                   "font-family: \"微软雅黑\";\n"
                                   "border-radius:12px;\n"
                                   "height:35px;\n"
                                   "}")
        self.frame_9.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_9.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_9.setObjectName("frame_9")
        self.verticalLayout_5 = QtWidgets.QVBoxLayout(self.frame_9)
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.id_lineEdit = QtWidgets.QLineEdit(self.frame_9)
        self.id_lineEdit.setObjectName("id_lineEdit")
        self.verticalLayout_5.addWidget(self.id_lineEdit)
        self.name_lineEdit = QtWidgets.QLineEdit(self.frame_9)
        self.name_lineEdit.setObjectName("name_lineEdit")
        self.verticalLayout_5.addWidget(self.name_lineEdit)
        self.password_lineEdit = QtWidgets.QLineEdit(self.frame_9)
        self.password_lineEdit.setObjectName("password_lineEdit")
        self.verticalLayout_5.addWidget(self.password_lineEdit)
        self.age_lineEdit = QtWidgets.QLineEdit(self.frame_9)
        self.age_lineEdit.setObjectName("age_lineEdit")
        self.verticalLayout_5.addWidget(self.age_lineEdit)
        self.sex_lineEdit = QtWidgets.QComboBox(self.frame_9)
        self.sex_lineEdit.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.sex_lineEdit.setStyleSheet("QComboBox{\n"
                                        "border:2px solid gray;\n"
                                        "font-size:16px;\n"
                                        "font-weight:600;\n"
                                        "font-family: \"微软雅黑\";\n"
                                        "border-radius:12px;\n"
                                        "height:35px;\n"
                                        "}\n"
                                        "\n"
                                        "QComboBox::drop-down { \n"
                                        "border: none;\n"
                                        "\n"
                                        "}")

        self.sex_lineEdit.setObjectName("sex_lineEdit")
        self.sex_lineEdit.addItem("")
        self.sex_lineEdit.addItem("")
        self.sex_lineEdit.addItem("")
        self.verticalLayout_5.addWidget(self.sex_lineEdit)
        self.more_lineEdit = QtWidgets.QLineEdit(self.frame_9)
        self.more_lineEdit.setObjectName("more_lineEdit")
        self.verticalLayout_5.addWidget(self.more_lineEdit)
        self.verticalLayout_4.addWidget(self.frame_9)
        self.verticalLayout_3.addWidget(self.frame_7)
        self.frame_8 = QtWidgets.QFrame(self.frame_4)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(2)
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

        Dialog.setAttribute(QtCore.Qt.WA_TranslucentBackground)  # 隐藏外围边框
        Dialog.setWindowFlag(QtCore.Qt.FramelessWindowHint)  # 产生一个无边框的窗口，用户不能移动和改变大小
        Dialog.setWindowTitle("人脸识别")  # 设置标题
        Dialog.setWindowIcon(QIcon(':/icons/images/dog.png'))  # 设置logo
        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.label_2.setText(_translate("Dialog", "基于人脸识别门禁系统"))
        self.id_lineEdit.setPlaceholderText(_translate("Dialog", "请输入学号"))
        self.name_lineEdit.setPlaceholderText(_translate("Dialog", "请输入姓名"))
        self.password_lineEdit.setPlaceholderText(_translate("Dialog", "设置登录密码"))
        self.age_lineEdit.setPlaceholderText(_translate("Dialog", "请输入年龄"))
        self.sex_lineEdit.setItemText(0, _translate("Dialog", "请选择性别"))
        self.sex_lineEdit.setItemText(1, _translate("Dialog", "男"))
        self.sex_lineEdit.setItemText(2, _translate("Dialog", "女"))
        self.more_lineEdit.setPlaceholderText(_translate("Dialog", "更多信息"))
        self.confirm_button.setText(_translate("Dialog", "注册"))
        self.cancel_button.setText(_translate("Dialog", "取消"))


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    widgets = QtWidgets.QMainWindow()
    ui = RegisterMsgUi()
    ui.setupUi(widgets)
    widgets.show()
    sys.exit(app.exec_())
