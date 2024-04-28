# -*- coding: utf-8 -*-
import sys
import qtawesome
import img
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QMainWindow
from PyQt5.QtCore import Qt, QPoint, QRect
from PyQt5.QtGui import QIcon, QCursor, QMouseEvent


class MyMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = StartMainUi()
        self.ui.setupUi(self)
        self.initWindow()

    def initWindow(self):
        # 初始化窗口移动的相关变量
        self.isMoving = False
        self.oldPos = QPoint()
        self.titleBarRect = QRect(0, 0, self.width(), 30)  # 假设标题栏高度为30像素

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.LeftButton and self.titleBarRect.contains(event.pos()):
            self.isMoving = True
            self.oldPos = event.globalPos()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self.isMoving:
            delta = event.globalPos() - self.oldPos
            self.move(self.x() + delta.x(), self.y() + delta.y())
            self.oldPos = event.globalPos()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        self.isMoving = False


class StartMainUi(object):
    def setupUi(self, MainWindow):
        MainWindow.resize(700, 450)
        self.main_widget = QtWidgets.QWidget(MainWindow)  # 创建窗口主部件
        self.main_layout = QtWidgets.QGridLayout()  # 创建主部件的网格布局
        self.main_widget.setLayout(self.main_layout)  # 设置窗口主部件布局为网格布局
        MainWindow.setCentralWidget(self.main_widget)
        self.close_button = QtWidgets.QPushButton(MainWindow)
        self.close_button.setGeometry(QtCore.QRect(30, 30, 31, 21))
        self.minimize_button = QtWidgets.QPushButton(MainWindow)
        self.minimize_button.setGeometry(QtCore.QRect(90, 30, 31, 21))
        self.Title_label = QtWidgets.QLabel(MainWindow)
        self.Title_label.setGeometry(QtCore.QRect(170, 100, 360, 60))
        self.user_button = QtWidgets.QPushButton(MainWindow)
        self.user_button.setGeometry(QtCore.QRect(150, 210, 400, 40))
        self.admin_button = QtWidgets.QPushButton(MainWindow)
        self.admin_button.setGeometry(QtCore.QRect(150, 270, 400, 40))
        self.cancel_button = QtWidgets.QPushButton(MainWindow)
        self.cancel_button.setGeometry(QtCore.QRect(150, 330, 400, 40))

        # ------------------------------------ #
        self.close_button.setFixedSize(30, 30)
        self.minimize_button.setFixedSize(30, 30)

        self.close_button.setStyleSheet(
            '''QPushButton{background:#F76677;border-radius:12px;}QPushButton:hover{background:red;}''')
        self.minimize_button.setStyleSheet(
            '''QPushButton{background:#6DDF6D;border-radius:12px;}QPushButton:hover{background:green;}''')

        MainWindow.setAttribute(Qt.WA_TranslucentBackground)
        MainWindow.setWindowFlag(QtCore.Qt.FramelessWindowHint)
        MainWindow.setAutoFillBackground(True)
        MainWindow.setWindowTitle("人脸识别")
        MainWindow.setWindowIcon(QIcon('Logo.jpg'))

        buttons = [self.user_button, self.admin_button, self.cancel_button]
        for button in buttons:
            button.setStyleSheet('''
                QPushButton{
                    border:none;
                    color:white;
                    padding-left:5px;
                    padding-right:10px;
                    font-size:20px;
                    font-weight:700;
                    font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;}
                QPushButton:hover{ 
                    color:white;
                    border:1.5px solid #F0F7F4;
                    border-radius:10px;
                    background:#355a70;}''')

        self.main_widget.setStyleSheet('''
            QWidget{
            background-image: url(:/icons/images/bg.png);  
            background-repeat: no-repeat; /* 图片不重复 */  
            border-top-right-radius:15px;
            border-top-left-radius:15px;
            border-bottom-right-radius:15px;
            border-bottom-left-radius:15px;}''')
        self.minimize_button.setCursor(QCursor(Qt.PointingHandCursor))
        self.close_button.setCursor(QCursor(Qt.PointingHandCursor))
        self.user_button.setCursor(QCursor(Qt.PointingHandCursor))
        self.cancel_button.setCursor(QCursor(Qt.PointingHandCursor))
        self.admin_button.setCursor(QCursor(Qt.PointingHandCursor))
        self.Title_label.setAlignment(Qt.AlignCenter)
        self.Title_label.setStyleSheet("QLabel{color:#F8FCFF;font-size:45px;font-weight:bold;font-family:Roman times;}")
        MainWindow.setWindowIcon(QIcon(':/icons/images/logo.jpg'))  # 设置logo
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.close_button.setText(_translate("Dialog", ""))
        self.close_button.setIcon(qtawesome.icon('fa.times', color='white'))
        # self.other_button.setText(_translate("Dialog", ""))
        self.minimize_button.setText(_translate("Dialog", ""))
        self.minimize_button.setIcon(qtawesome.icon('fa.minus', color='white'))
        self.Title_label.setText(_translate("Form", "--图书馆门禁系统--"))
        self.user_button.setText(_translate("Form", "使用用户身份"))
        self.admin_button.setText(_translate("Form", "使用管理员身份"))
        self.cancel_button.setText(_translate("Form", "退出"))


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    mainWindow = MyMainWindow()
    mainWindow.show()
    sys.exit(app.exec_())
