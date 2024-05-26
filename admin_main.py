# -*- coding: utf-8 -*-

import sys
import qtawesome
import pandas as pd
import os
import torch
import shutil
import cv2
import numpy as np
import configparser
import torchvision.transforms as transforms
from facenet_pytorch import MTCNN, InceptionResnetV1
from PyQt5.QtWidgets import QApplication, QMainWindow, QDialog, QMessageBox, QProgressBar, QTableWidgetItem, \
    QTableWidget
from PyQt5.QtCore import Qt, QPoint, QSize, QPropertyAnimation, QEasingCurve, pyqtSlot, QTimer
from PyQt5.QtGui import QKeyEvent, QPixmap
from PyQt5 import uic, QtWidgets
from util.AdminSqlUtilManager import AdminConnect
from util.share import SharedInformation
from util import UserSqlUtil
from PIL import Image
from datetime import datetime


# 登录窗口
class WinLoginUi(QMainWindow):
    def __init__(self):
        # 初始化 QMainWindow
        super().__init__()
        # 在 QMainWindow 实例上加载 UI
        uic.loadUi("ui/admin_login.ui", self)
        # 设置窗口标志以隐藏标题栏和外围边框
        self.setWindowFlags(Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        # 设置窗口拖动相关的属性
        self.drag_position = QPoint()
        # 组件美化
        self.component_beautification()
        # ---------- 触发事件 ---------- #
        # 最小化和关闭按钮
        self.close_button.clicked.connect(self.close_window)
        self.minimize_button.clicked.connect(self.minimize_window)
        # 登录按钮按下
        self.login_button.clicked.connect(self.login_window)

    def close_window(self):
        self.close()

    def minimize_window(self):
        self.showMinimized()

    def component_beautification(self):
        # 关闭按钮美化
        icon_close = qtawesome.icon('mdi.close', color='#7b8290')
        self.close_button.setIcon(icon_close)
        self.close_button.setIconSize(QSize(30, 30))
        # 最小化按钮美化
        icon_minimize = qtawesome.icon('mdi.minus', color='#7b8290')
        self.minimize_button.setIcon(icon_minimize)
        self.minimize_button.setIconSize(QSize(30, 30))

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drag_position = event.globalPos() - self.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event):
        if not event.buttons() & Qt.LeftButton:
            return
        self.move(event.globalPos() - self.drag_position)
        event.accept()

    def mouseReleaseEvent(self, event):
        # 可选：实现鼠标释放事件的处理逻辑
        event.accept()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key_Return or event.key() == Qt.Key_Enter:
            self.on_enter_pressed()
        else:
            pass

    def on_enter_pressed(self):
        # 在这里处理 Enter 键被按下的逻辑
        self.login_window()

    def login_window(self):
        input_id = self.admin_lineEdit.text().strip()
        input_password = self.password_lineEdit.text().strip()

        if input_id == "":
            QMessageBox.information(self, '提示', '姓名不能为空')
        elif input_password == "":
            QMessageBox.information(self, '提示', '密码不能为空')
        else:
            row = AdminConnect().search_by_id(input_id)
            if row:
                result = row[0]
                password = result[1]
                if input_password != password:
                    QMessageBox.information(self, '提示', '密码输入错误')
                else:
                    # 加载进度条动画
                    self.show_loading_progress()
            else:
                QMessageBox.information(self, '提示', '该用户不存在')

    def show_loading_progress(self):
        self.hide()
        SharedInformation.progressBar = AnimatedProgressBar(500)
        SharedInformation.progressBar.exec_()
        SharedInformation.mainWin = WinMainUi()
        SharedInformation.mainWin.show()


# 管理员主窗口
class WinMainUi(QMainWindow):
    def __init__(self):
        # 初始化 QMainWindow
        super().__init__()
        # 在 QMainWindow 实例上加载 UI
        uic.loadUi("ui/admin_main_window.ui", self)
        # 设置窗口标志以隐藏标题栏和外围边框
        self.setWindowFlags(Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        # 设置窗口拖动相关的属性
        self.drag_position = QPoint()
        # 组件美化
        self.component_beautification()
        # ---------- 定义 ---------- #
        self.all_list = []  # 列表
        self.isSearch_flag = 0
        self.hasModify_flag = 0
        self.sid = None
        self.name = None
        self.password = None
        self.age = None
        self.sex = None
        self.more = None
        path_1 = './data/successful_identification_records.xlsx'
        path_2 = './data/login_records.xlsx'
        # ---------- 触发事件 ---------- #
        # 最小化和关闭按钮
        self.close_button.clicked.connect(self.close_window)
        self.minimize_button.clicked.connect(self.minimize_window)
        # 初始化表格
        self.init_tables(path_1, path_2)
        self.fill_table_widget_with_data()
        # 页面切换
        self.information_button.clicked.connect(lambda: self.on_information_clicked(0))
        self.record_button.clicked.connect(lambda: self.on_information_clicked(1))
        self.login_button.clicked.connect(lambda: self.on_information_clicked(2))
        # 返回登录界面按钮
        self.back_button.clicked.connect(self.back_login_win)
        # 两个记录下面的按钮事件
        # 删除数据按钮
        self.del_button_tab1.clicked.connect(lambda: self.delete_selected_row(self.table_widget_success))
        self.del_button_tab2.clicked.connect(lambda: self.delete_selected_row(self.table_widget_login))
        # 增加数据按钮
        self.add_button_tab1.clicked.connect(lambda: self.add_new_row(self.table_widget_success))
        self.add_button_tab2.clicked.connect(lambda: self.add_new_row(self.table_widget_login))
        # 保存数据按钮
        self.save_button_tab1.clicked.connect(lambda: self.save_table_to_excel(self.table_widget_success, path_1))
        self.save_button_tab2.clicked.connect(lambda: self.save_table_to_excel(self.table_widget_login, path_2))
        # 导出数据
        self.successful_button_tab1.clicked.connect(lambda: self.export_table_to_excel(self.table_widget_success))
        self.successful_button_tab2.clicked.connect(lambda: self.export_table_to_excel(self.table_widget_login))
        # ------ table_widget_student部分 ------ #
        # 设置table_widget_student选择行为整行选择
        self.table_widget_student.setSelectionBehavior(QTableWidget.SelectRows)
        # 查询按钮事件
        self.student_search_button.clicked.connect(self.on_search_button_student_clicked)
        # 重置按钮事件
        self.reset_button.clicked.connect(lambda: self.reset_table_widget(self.table_widget_student))
        # 修改按钮事件
        self.modify_button.clicked.connect(self.modify_button_clicked)
        # 查看照片事件
        self.student_photo_button.clicked.connect(self.view_photo)
        # 输入框居中显示
        line_text = [self.change_id_lineEdit, self.name_lineEdit, self.passsword_lineEdit, self.age_lineEdit,
                     self.sex_lineEdit, self.more_lineEdit]
        for lineEdit in line_text:
            lineEdit.setAlignment(Qt.AlignCenter)
        # 确认修改事件
        self.confirm_button.clicked.connect(self.confirm_button_clicked)
        # 删除数据事件
        self.del_student_button.clicked.connect(self.del_student_button_clicked)
        # 添加学生事件
        self.register_student_button.clicked.connect(self.register_button_clicked)
        # 设置窗口
        self.setting_button.clicked.connect(self.setting_win_open)
        # 关于本软件
        self.about_button.clicked.connect(self.about_win_open)
        # 退出登录事件
        self.loginout_button.clicked.connect(self.back_login_win)

    def close_window(self):
        # 弹出提示框
        reply = QMessageBox.question(self, '关闭窗口',
                                     "你确定退出吗?", QMessageBox.Yes |
                                     QMessageBox.No, QMessageBox.No)
        # 检查用户的响应
        if reply == QMessageBox.Yes:
            self.close()

    def minimize_window(self):
        self.showMinimized()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drag_position = event.globalPos() - self.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event):
        if not event.buttons() & Qt.LeftButton:
            return
        self.move(event.globalPos() - self.drag_position)
        event.accept()

    def mouseReleaseEvent(self, event):
        # 可选：实现鼠标释放事件的处理逻辑
        event.accept()

    def back_login_win(self):
        self.close()
        SharedInformation.loginWin.show()

    def component_beautification(self):
        # 关闭按钮美化
        icon_close = qtawesome.icon('mdi.close', color='grey')
        self.close_button.setIcon(icon_close)
        self.close_button.setIconSize(QSize(30, 30))
        # 最小化按钮美化
        icon_minimize = qtawesome.icon('mdi.minus', color='grey')
        self.minimize_button.setIcon(icon_minimize)
        self.minimize_button.setIconSize(QSize(30, 30))
        # 其他组件美化
        self.information_button.setIcon(qtawesome.icon('mdi6.information-box', color='#787f8d'))
        self.record_button.setIcon(qtawesome.icon('ph.record-fill', color='#787f8d'))
        self.login_button.setIcon(qtawesome.icon('mdi.login-variant', color='#787f8d'))
        self.setting_button.setIcon(qtawesome.icon('ri.settings-4-fill', color='#787f8d'))
        self.about_button.setIcon(qtawesome.icon('mdi6.white-balance-sunny', color='#787f8d'))
        self.loginout_button.setIcon(qtawesome.icon('fa.sign-out', color='#787f8d'))

    def init_tables(self, path_1, path_2):
        """初始化并填充表格"""
        self.fill_table_widget(path_1, self.table_widget_success)
        self.fill_table_widget(path_2, self.table_widget_login)

    @staticmethod
    def fill_table_widget(file_path, table_widgets):
        # 读取Excel文件
        df = pd.read_excel(file_path)
        # 清除旧数据（如果有的话）
        table_widgets.clear()
        table_widgets.setRowCount(0)
        table_widgets.setColumnCount(len(df.columns))
        table_widgets.setHorizontalHeaderLabels(df.columns)
        # 设置表格的行数
        table_widgets.setRowCount(len(df))
        # 填充数据
        for row_index, row in df.iterrows():
            for column_index, value in enumerate(row):
                item = QtWidgets.QTableWidgetItem(str(value))
                item.setTextAlignment(Qt.AlignCenter)  # 设置内容为居中
                table_widgets.setItem(row_index, column_index, item)
        table_widgets.resizeColumnToContents(0)

    def read_person_msg(self):
        """读取用户信息"""
        all_msg = UserSqlUtil.search_all_msg()
        for tup in all_msg:
            per_list = []
            for i in tup:
                per_list.append(i)
            self.all_list.append(per_list)

    def fill_table_widget_with_data(self):
        """读取数据并填充 self.all_list"""
        self.read_person_msg()
        # 如果没有数据，则不执行任何操作
        if not self.all_list:
            return
        self.table_widget_student.setRowCount(0)
        self.traverse_list(self.table_widget_student)

    def traverse_list(self, table_widget):
        """遍历所有的元组（即查询结果）"""
        for tup in self.all_list:
            # 提取前三个元素
            per_list = tup[:6]
            # 清空表格的现有行
            # 添加一行到表格中
            row_count = table_widget.rowCount()
            table_widget.insertRow(row_count)
            # 将每个元素添加到对应的单元格中
            for column_index, value in enumerate(per_list):
                item = QTableWidgetItem(str(value))  # 将值转换为字符串并创建 QTableWidgetItem
                item.setTextAlignment(Qt.AlignCenter)  # 设置内容为居中
                table_widget.setItem(row_count, column_index, item)  # 将 item 添加到表格中
        self.all_list = []

    def on_information_clicked(self, index):
        """切换到StackedWidget的页面"""
        self.stackedWidget.setCurrentIndex(index)

    def delete_selected_row(self, table_widgets):
        """删除按钮事件"""
        # 获取当前选中的行
        selected_rows = table_widgets.selectedItems()
        if not selected_rows:
            QtWidgets.QMessageBox.information(self, "提示", "请选中要删除数据")
            return
        # 询问用户是否真的要删除选定的行
        reply = QMessageBox.question(self, '确定吗', '您确定要删除当前选定的的数据行吗?',
                                     QMessageBox.Yes | QMessageBox.No,
                                     QMessageBox.No)
        # 如果用户点击了"是"
        if reply == QMessageBox.Yes:
            # 获取选中的行索引
            selected_row_indexes = set()
            for item in selected_rows:
                selected_row_indexes.add(table_widgets.row(item))
                # 对选中的行进行降序排序，以确保从底部开始删除
            selected_row_indexes = sorted(selected_row_indexes, reverse=True)
            # 删除选中的行
            for row_index in selected_row_indexes:
                table_widgets.removeRow(row_index)

    @staticmethod
    def add_new_row(table_widgets):
        # 获取当前选中的行索引
        selected_row = table_widgets.currentRow()
        # 如果没有行被选中，则在表格底部添加一行
        if selected_row == -1:
            row_count = table_widgets.rowCount()
            table_widgets.insertRow(row_count)
        else:
            # 在选中的行下面添加一行
            table_widgets.insertRow(selected_row + 1)
            # 初始化新行的每个单元格
        for column_index in range(table_widgets.columnCount()):
            item = QtWidgets.QTableWidgetItem("")
            item.setTextAlignment(Qt.AlignCenter)  # 设置内容为居中
            table_widgets.setItem(selected_row + 1, column_index, item)
        # 滚动到底部以便看到新行（如果需要）
        table_widgets.scrollToItem(table_widgets.item(selected_row + 1, 0))

    # 保存表格修改的内容
    def save_table_to_excel(self, table_widgets, file_path):
        # 获取表格的行数和列数
        rows = table_widgets.rowCount()
        columns = table_widgets.columnCount()

        # 获取水平表头的标签
        header_labels = [table_widgets.horizontalHeaderItem(i).text() for i in range(columns)]

        # 创建一个空的DataFrame，其列与表格的列对应
        df = pd.DataFrame(index=range(rows), columns=header_labels)

        # 遍历表格的每一行和每一列
        for row in range(rows):
            for column in range(columns):
                # 获取单元格的项
                item = table_widgets.item(row, column)
                # 如果单元格有项，则获取其文本并保存到DataFrame中
                # 如果单元格没有项（即None），则保存一个空字符串
                df.iat[row, column] = item.text() if item is not None else ''
        # 将DataFrame保存为Excel文件
        try:
            # 写入 Excel 文件
            df.to_excel(file_path, index=False)

            # 检查文件是否存在
            if os.path.exists(file_path):
                QMessageBox.information(self, "提示", f'数据保存成功')
            else:
                QMessageBox.information(self, "错误", '文件未创建，数据写入失败。')

        except Exception as e:
            # 捕获并打印异常
            QMessageBox.information(self, "错误", f"数据写入过程中发生异常: {e}")

    @staticmethod
    def export_table_to_excel(table_widgets):
        # 收集表格数据到列表中，导出表格
        rows = []
        for row_index in range(table_widgets.rowCount()):
            row_data = []
            for column_index in range(table_widgets.columnCount()):
                item = table_widgets.item(row_index, column_index)
                if item is not None:
                    row_data.append(item.text())
                else:
                    row_data.append('')  # 如果没有数据，添加空字符串
            rows.append(row_data)
        # 创建DataFrame
        df = pd.DataFrame(rows, columns=[table_widgets.horizontalHeaderItem(i).text() for i in
                                         range(table_widgets.columnCount())])
        file_path = 'example.xlsx'
        try:
            # 将DataFrame保存为Excel文件
            df.to_excel(file_path, index=False)
            print(f"Excel文件已保存到: {file_path}")
        except Exception as e:
            print(f"创建Excel文件时出错: {e}")
        if os.path.isfile(file_path):
            # 根据操作系统选择打开文件的方式
            if sys.platform.startswith('win'):
                os.startfile(file_path)  # Windows系统
        else:
            # 如果文件不存在，显示错误消息
            print("文件不存在")

    def on_search_button_student_clicked(self):
        # 获取 id_lineEdit 的内容
        search_id = self.id_lineEdit.text()
        if search_id == '':
            return QMessageBox.information(self, "提示", "学号不能为空，请输入学号")
        # 假设 search_by_name 方法根据名称查询并返回对应的行，这里应该根据实际需求调整方法名和逻辑
        row = UserSqlUtil.search_by_name(search_id)
        self.table_widget_student.setRowCount(0)
        if row:
            # row 是一个元组，我们取第一个元素（通常是一个包含所有列数据的元组）
            result = row[0]
            # 清空表格的现有行
            # 添加一行到表格中
            row_count = self.table_widget_student.rowCount()
            self.table_widget_student.insertRow(row_count)
            # 将 result 的前四个元素添加到表格的对应单元格中
            for column_index, value in enumerate(result[:6]):
                item = QTableWidgetItem(str(value))  # 将值转换为字符串并创建 QTableWidgetItem
                item.setTextAlignment(Qt.AlignCenter)  # 设置内容为居中
                self.table_widget_student.setItem(row_count, column_index, item)  # 将 item 添加到表格中
        else:
            QMessageBox.about(self, '提示', '找不到' + search_id + '的信息')

    def reset_table_widget(self, table_widget):
        self.read_person_msg()
        # 清空表格中的所有行
        table_widget.setRowCount(0)
        # 遍历所有的元组（即查询结果）
        self.traverse_list(table_widget)

    def modify_button_clicked(self):
        # 获取 id_lineEdit 的内容
        search_id = self.id_lineEdit.text()
        if search_id == '':
            return QMessageBox.information(self, "提示", "学号不能为空，请输入学号")
        # 假设 search_by_name 方法根据名称查询并返回对应的行，这里应该根据实际需求调整方法名和逻辑
        row = UserSqlUtil.search_by_name(search_id)
        if row:
            result = row[0]
            self.sid = result[0]
            self.name = result[1]
            self.password = result[2]
            self.age = result[3]
            self.sex = result[4]
            self.more = result[5]
            self.change_id_lineEdit.setText("")
            self.name_lineEdit.setText("")
            self.passsword_lineEdit.setText("")
            self.age_lineEdit.setText("")
            self.sex_lineEdit.setText("")
            self.more_lineEdit.setText("")
            self.change_id_lineEdit.setPlaceholderText(str(self.sid))
            self.name_lineEdit.setPlaceholderText(str(self.name))
            self.passsword_lineEdit.setPlaceholderText(str(self.password))
            self.age_lineEdit.setPlaceholderText(str(self.age))
            self.sex_lineEdit.setPlaceholderText(str(self.sex))
            self.more_lineEdit.setPlaceholderText(str(self.more))
            self.isSearch_flag = 1
        else:
            QMessageBox.about(self, '提示', '找不到' + search_id + '的信息')

    def confirm_button_clicked(self):
        if self.isSearch_flag == 1:
            sid = self.change_id_lineEdit.text()
            name = self.name_lineEdit.text()
            password = self.passsword_lineEdit.text()
            age = self.age_lineEdit.text()
            sex = self.sex_lineEdit.text()
            more = self.more_lineEdit.text()
            if sid != '':
                self.hasModify_flag = 1
            else:
                sid = self.sid
            if password != '':
                self.hasModify_flag = 1
            else:
                password = self.password
            if name != '':
                self.hasModify_flag = 1
            else:
                name = self.name
            if age != '' and str.isdigit(age) is True:
                self.hasModify_flag = 1
            else:
                age = self.age
            if sex != '' and (sex == '男' or sex == '女'):
                self.hasModify_flag = 1
            else:
                sex = self.sex
            if more != '':
                self.hasModify_flag = 1
            else:
                more = self.more
            if self.hasModify_flag == 1:
                UserSqlUtil.update_by_id_without_encoding(sid, name, password, age, sex, more)
                self.isSearch_flag = 0
                QMessageBox.about(self, '更新', str(sid) + '的部分信息已更改')
            else:
                QMessageBox.about(self, '提示', '请先修改信息')
        else:
            QMessageBox.about(self, '提示', '请先输入正确的学号，单击修改数据按钮后再修改数据！')

    def view_photo(self):
        if self.table_widget_student.selectedItems():
            # 获取第一个选中项的行号
            row_table = self.table_widget_student.selectedItems()[0].row()
            # 遍历该行的所有列
            row_datas = []
            for column in range(self.table_widget_student.columnCount()):
                item = self.table_widget_student.item(row_table, column)
                if item:
                    row_datas.append(item.text())
                else:
                    row_datas.append('')
            # 打印选中行的内容
            table_id = row_datas[0]
            row = UserSqlUtil.search_by_name("\"" + table_id + "\"")
            if row:
                result = row[0]
                sid = result[0]
                name = result[1]
                password = result[2]
                age = result[3]
                sex = result[4]
                more = result[5]
                image = result[7]
                SharedInformation.userShowWin = UserShowWin(sid, name, password, age, sex, more, image)
                SharedInformation.userShowWin.exec_()
            else:
                QMessageBox.about(self, '提示', '找不到' + table_id + '的信息')
        else:
            QMessageBox.information(self, '提示', '请选择表格中的数据!!!')

    def del_student_button_clicked(self):
        # 检查是否有选中的行
        if self.table_widget_student.selectedItems():
            # 获取第一个选中项的行号
            row = self.table_widget_student.selectedItems()[0].row()
            # 遍历该行的所有列
            row_data = []
            for column in range(self.table_widget_student.columnCount()):
                item = self.table_widget_student.item(row, column)
                if item:
                    row_data.append(item.text())
                else:
                    row_data.append('')
            # 打印选中行的内容
            table_id = row_data[0]
            # 假设 table_id 方法根据名称查询并返回对应的行，这里应该根据实际需求调整方法名和逻辑
            row = UserSqlUtil.search_by_name(table_id)
            if row:
                reply = QMessageBox.question(self, '确定吗', f'您确定要删除用户为：{table_id} 的所有数据吗？',
                                             QMessageBox.Yes | QMessageBox.No,
                                             QMessageBox.No)
                # 如果用户点击了"是"
                if reply == QMessageBox.Yes:
                    flag = UserSqlUtil.delete_by_name(table_id)
                    if flag:
                        QMessageBox.about(self, '提示', f'用户{table_id}删除成功,请重置表格数据查看')
                    else:
                        QMessageBox.about(self, '提示', f'用户{table_id}删除失败，请重试')
            else:
                QMessageBox.about(self, '提示', f'用户{table_id}不存在')
        else:
            QMessageBox.information(self, '提示', '请选择表格中的数据，再删除')

    def register_button_clicked(self):
        QTimer.singleShot(0, self.progress_bar_update)  # 5秒后关闭
        QApplication.processEvents()
        SharedInformation.imgWin = ImgUi()
        SharedInformation.imgWin.exec_()

    @staticmethod
    def progress_bar_update():
        SharedInformation.progressBar = AnimatedProgressBar(1000)
        SharedInformation.progressBar.exec_()

    @staticmethod
    def about_win_open():
        SharedInformation.aboutWin = AboutDialog()
        SharedInformation.aboutWin.exec_()

    @staticmethod
    def setting_win_open():
        SharedInformation.settingWin = SettingsWin()
        SharedInformation.settingWin.exec_()


# 进程动画
class AnimatedProgressBar(QDialog):
    def __init__(self, duration=None):
        # 初始化 QMainWindow
        super().__init__()
        # 在 QMainWindow 实例上加载 UI
        uic.loadUi("ui/progress_bar.ui", self)
        # 设置窗口标志以隐藏标题栏和外围边框
        self.setWindowFlags(Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        # 获取进度条对象
        self.progress_bar = self.findChild(QProgressBar, "main_progress")
        self.duration = duration
        if self.progress_bar is None:
            print("Error: Unable to find the progress bar with objectName 'main_progress'")
            return

            # 设置进度条的范围
        self.progress_bar.setRange(0, 100)

        # 初始化动画
        self.animation = QPropertyAnimation(self.progress_bar, b"value")
        self.animation.setDuration(self.duration)  # 持续时间 3 秒
        self.animation.setStartValue(0)
        self.animation.setEndValue(100)
        self.animation.setEasingCurve(QEasingCurve.Linear)
        self.animation.finished.connect(self.close)
        self.show()
        self.animation.start()


# 选择图片窗口
class ImgUi(QDialog):
    def __init__(self, device=None):
        # 初始化 QMainWindow
        super().__init__()
        # 在 QMainWindow 实例上加载 UI
        uic.loadUi("ui/select_image.ui", self)
        # ------- 模型配置 ------- #
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        self.mtcnn = MTCNN(keep_all=True, device=self.device)
        # 初始化InceptionResnetV1模型，用于特征提取
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        self.min_face_size = 120
        self.numpy_features = []
        self.isPhoto = 0
        self.img_name = None
        self.processed_frame = None
        # ------- 窗口其他 ------- #
        # 设置窗口标志以隐藏标题栏和外围边框
        self.setWindowFlags(Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        # 设置窗口拖动相关的属性
        self.drag_position = QPoint()
        # 组件美化
        self.component_beautification()
        # 设置 QLabel 可以接收鼠标事件
        self.image_label.mousePressEvent = self.on_label_clicked
        # 确认按钮按下
        self.ok_button.clicked.connect(self.ok_button_clicked)
        self.close_button.clicked.connect(self.close_clicked)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key_Return or event.key() == Qt.Key_Enter:
            self.on_enter_pressed()
        else:
            pass

    def on_enter_pressed(self):
        # 在这里处理 Enter 键被按下的逻辑
        self.ok_button_clicked()

    def close_clicked(self):
        self.close()
        SharedInformation.mainWin.show()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drag_position = event.globalPos() - self.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event):
        if not event.buttons() & Qt.LeftButton:
            return
        self.move(event.globalPos() - self.drag_position)
        event.accept()

    def mouseReleaseEvent(self, event):
        event.accept()

    def component_beautification(self):
        # 关闭按钮美化
        icon_close = qtawesome.icon('mdi.close', color='grey')
        self.close_button.setIcon(icon_close)
        self.close_button.setIconSize(QSize(30, 30))

    @pyqtSlot()
    def on_label_clicked(self, event):
        """检查是否是左键点击"""
        if event.button() == Qt.LeftButton:
            self.open_image()

    def open_image(self):
        """使用默认样式的QFileDialog打开图片"""
        file_name, _ = QtWidgets.QFileDialog.getOpenFileName(None, "选择图片", "",
                                                             "Images (*.PNG *.JPG *.BMP *.JPEG)")
        if file_name:
            frame = cv2.imread(file_name)
            if frame is None:
                QMessageBox.information(self, "错误", "无法读取图片，请检查路径是否正确")
                return
            if len(frame.shape) == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                # 检测人脸
            self.processed_frame, boxes = self.detect_faces(frame)
            # 如果没有检测到人脸，则显示错误信息
            if boxes is None or len(boxes) == 0:
                QMessageBox.information(self, "错误", "没有人脸检测到，请检查图片")
            else:
                # 提取人脸特征
                face_features = self.extract_face_features(frame)
                if face_features is not None:
                    self.numpy_features = self.convert_features_to_array(face_features)
                    # 显示信息
                    QMessageBox.information(self, "提示", "检测到人脸!")

                    pixmap = QPixmap(file_name)
                    self.image_label.setPixmap(pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio))
                    self.isPhoto = 1
                    self.img_name = file_name
                else:
                    self.isPhoto = 0
                    QMessageBox.information(self, "提示", "没有检测到人脸！")

    def detect_faces(self, frame):
        boxes, _ = self.mtcnn.detect(frame)
        filtered_boxes = []
        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = box  # 假设detect方法返回的每个box包含置信度
                width = x2 - x1
                height = y2 - y1
                face_size = min(width, height)  # 计算人脸尺寸
                # 过滤小于最小尺寸阈值的人脸
                if face_size >= self.min_face_size:
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    filtered_boxes.append(box)  # 将符合条件的box添加到列表中
        # 返回修改后的frame和过滤后的人脸框列表
        return frame, filtered_boxes

    def extract_face_features(self, frame):
        # 检测人脸并获取边界框
        frame_with_boxes, boxes = self.detect_faces(frame.copy())
        # 初始化一个空的特征列表，用于存储当前帧的特征
        face_features = []
        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = box
                # 根据边界框裁剪人脸区域
                face = frame[int(y1):int(y2), int(x1):int(x2)]
                # 调整图像大小以匹配模型的输入要求，并转换为Tensor
                face_tensor = self._prepare_face_tensor(face)
                if face_tensor is not None:
                    # 使用预训练的模型提取特征
                    with torch.no_grad():
                        feature = self.resnet(face_tensor).cpu()
                        face_features.append(feature)
                    return face_features
                else:
                    return None

    def _prepare_face_tensor(self, face):
        # 将裁剪出的人脸图像转换为模型所需的Tensor格式
        if face.size == 0:
            print("Warning: Empty face image, skipping tensor preparation.")
            return None  # 或者返回其他适当的值，比如一个全零的Tensor
        else:
            face = cv2.resize(face, (160, 160))  # 调整图像大小以适应模型输入
            face = face.astype(np.float32) / 255.0  # 归一化图像
            face = face.transpose((2, 0, 1))  # 转换图像维度顺序
            face_tensor = torch.from_numpy(face).unsqueeze(0).to(self.device)  # 转换为Tensor并添加到设备
            return face_tensor

    def convert_features_to_array(self, face_features):
        for feature in face_features:
            # 将PyTorch张量转换为NumPy数组，并添加到列表中
            numpy_feature = feature.squeeze().cpu().numpy()
            self.numpy_features.append(numpy_feature)
        return self.numpy_features

    def ok_button_clicked(self):
        if self.isPhoto == 1:
            self.hide()
            height, width = self.processed_frame.shape[:2]
            resized_frame = cv2.resize(self.processed_frame, (width // 4, height // 4))
            # 现在显示调整尺寸后的图像
            cv2.imshow('Face Recognitions', resized_frame)
            QApplication.processEvents()
            self.registration_dialog_box()
        else:
            QMessageBox.warning(self, "Warning", "请上传照片后再进行注册！")

    def registration_dialog_box(self):
        SharedInformation.registerWin = RegisterStudentUi(self.img_name)
        SharedInformation.registerWin.exec_()


# 注册窗口
class RegisterStudentUi(QDialog):
    def __init__(self, img_file_name=None, device=None):
        super().__init__()
        # 在 QMainWindow 实例上加载 UI
        uic.loadUi("ui/admin_user_register.ui", self)
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        self.mtcnn = MTCNN(keep_all=True, device=self.device)
        # 初始化InceptionResnetV1模型，用于特征提取
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        self.img_file_name = img_file_name
        print(img_file_name)
        self.flag = 0
        # 参数设置
        self.face_transformer = transforms.Compose([
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        # 设置窗口标志以隐藏标题栏和外围边框
        self.setWindowFlags(Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        # 设置窗口拖动相关的属性
        self.drag_position = QPoint()
        # 组件美化
        self.component_beautification()
        # 关闭按钮
        self.close_button.clicked.connect(self.close_up)
        # 取消按钮
        self.cancel_button.clicked.connect(self.close_up)
        # 注册按钮按下
        self.confirm_button.clicked.connect(self.fill_information)
        self.project_root_path = os.path.abspath(os.path.dirname(__file__))

        # 图片文件夹路径，相对于项目根路径
        self.photo_folder_name = 'photo'
        self.photo_folder_path = os.path.join(self.project_root_path, self.photo_folder_name)
        # 临时变量
        self.photo_name = None
        self.target_file_path = ''

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key_Return or event.key() == Qt.Key_Enter:
            self.on_enter_pressed()
        else:
            pass

    def on_enter_pressed(self):
        self.fill_information()

    def close_up(self):
        cv2.destroyWindow('Face Recognitions')
        self.close()

    def component_beautification(self):
        # 关闭按钮美化
        icon_close = qtawesome.icon('mdi.close', color='grey')
        self.close_button.setIcon(icon_close)
        self.close_button.setIconSize(QSize(30, 30))

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drag_position = event.globalPos() - self.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event):
        if not event.buttons() & Qt.LeftButton:
            return
        self.move(event.globalPos() - self.drag_position)
        event.accept()

    def mouseReleaseEvent(self, event):
        # 可选：实现鼠标释放事件的处理逻辑
        event.accept()

    # 注册事件
    def fill_information(self):
        sid = self.id_lineEdit.text()
        name = self.name_lineEdit.text()
        password = self.password_lineEdit.text()
        age = self.age_lineEdit.text()
        sex = self.sex_lineEdit.currentText()
        more_infor = self.more_lineEdit.text()

        if self.judge_name_conflict(sid):
            if sid != '':
                # 输入密码
                if password != '':
                    # 输入姓名
                    if name == '':
                        name = '未知'
                    elif not str.isdigit(age):
                        self.flag = 1
                        QMessageBox.about(self, '提示', '请输入姓名')
                    # 输入年龄
                    if age == '':
                        age = '未知'
                    elif not str.isdigit(age):
                        self.flag = 1
                        QMessageBox.about(self, '提示', '请输入正确的年龄格式')
                    # 输入性别
                    if sex == '':
                        sex = '未知'
                    elif sex != '男' and sex != '女':
                        self.flag = 1
                        QMessageBox.about(self, '提示', '请输入正确的性别格式')
                        sex = '未知'
                    # 输入更多信息
                    if more_infor == '':
                        more_infor = '未知'
                    if self.flag == 0:
                        # 计算脸部数据并保存到数据库
                        QApplication.processEvents()
                        register_encoding = self.analyse_encoding(self.img_file_name)
                        self.copy_image_to_folder()
                        image_path = self.target_file_path
                        if self.save_database(sid, name, password, age, sex, more_infor, register_encoding,
                                              image_path):
                            QMessageBox.about(self, '提示', '完成注册')
                            cv2.destroyWindow('Face Recognitions')
                            self.close()
                        else:
                            QMessageBox.about(self, '提示', '注册失败')
                            os.remove(image_path)
                    elif self.flag == 1:
                        QMessageBox.about(self, '提示', '注册失败')
                else:
                    QMessageBox.about(self, '提示', '请输入密码')
            else:
                QMessageBox.about(self, '提示', '请输入学号')
        else:
            QMessageBox.about(self, '提示', '用户' + sid + '已经注册过')

    def copy_image_to_folder(self):
        # 确保图片文件夹存在
        if not os.path.exists(self.photo_folder_path):
            os.makedirs(self.photo_folder_path)

            # 创建一个包含当前时间戳的文件名
        current_time = datetime.now().strftime("%Y%m%d%H%M%S")
        # 假设原始图片没有扩展名，或者您想要忽略它，则可以直接使用当前时间作为文件名
        # 否则，您可能需要从原始文件名中提取扩展名并添加到新文件名中
        extension = os.path.splitext(self.img_file_name)[-1]  # 获取文件扩展名
        target_file_name = f"{current_time}{extension}"  # 拼接文件名和扩展名
        # 构造目标文件路径
        self.target_file_path = os.path.join(self.photo_folder_path, target_file_name)
        # 复制图片到指定文件夹
        shutil.copy2(self.img_file_name, self.target_file_path)  # 使用 copy2 来保留元数据

    @staticmethod
    def judge_name_conflict(sid):
        count = UserSqlUtil.search_count_name("\"" + sid + "\"")
        if count != 0:
            return False
        else:
            return True

    # 分析截图
    def numpy_face_features(self, image_path):
        image = Image.open(image_path)
        boxes, probs, points = self.mtcnn.detect(image, landmarks=True)
        if boxes is None:
            raise ValueError("No face detected in the image.")
        box = boxes[0]
        face = image.crop(box).resize((160, 160))
        face_tensor = self.face_transformer(face).unsqueeze(0).to(self.device)
        with torch.no_grad():
            face_features = self.resnet(face_tensor)
        return face_features.cpu().numpy()

    def analyse_encoding(self, img_file_name):
        photo_path = img_file_name
        face_features_numpy = self.numpy_face_features(photo_path)
        face_encodings_binary = face_features_numpy.tobytes()
        return face_encodings_binary

    # 保存注册信息
    @staticmethod
    def save_database(sid, name, password, age, sex, more, face_encoding, image_path):
        return UserSqlUtil.insert_data(sid, name, password, age, sex, more, face_encoding, image_path)


# 关于软件窗口
class AboutDialog(QDialog):
    def __init__(self):
        # 初始化 QMainWindow
        super().__init__()
        # 在 QMainWindow 实例上加载 UI
        uic.loadUi("ui/about_software.ui", self)
        # 设置窗口标志以隐藏标题栏和外围边框
        self.setWindowFlags(Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        # 设置窗口拖动相关的属性
        self.drag_position = QPoint()
        self.confirm_button.clicked.connect(self.about)
        # 关闭窗口美化
        icon_close = qtawesome.icon('mdi.close', color='#7b8290')
        self.close_button.setIcon(icon_close)
        self.close_button.setIconSize(QSize(30, 30))
        # 关闭窗口事件
        self.close_button.clicked.connect(self.close)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drag_position = event.globalPos() - self.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event):
        if not event.buttons() & Qt.LeftButton:
            return
        self.move(event.globalPos() - self.drag_position)
        event.accept()

    def mouseReleaseEvent(self, event):
        event.accept()

    def about(self):
        QMessageBox.about(self, "About", "已经是最新版本！")


# 管理设置窗口
class SettingsWin(QDialog):
    def __init__(self):
        # 初始化 QMainWindow
        super().__init__()
        # 在 QMainWindow 实例上加载 UI
        uic.loadUi("ui/management_settings.ui", self)
        # 设置窗口标志以隐藏标题栏和外围边框
        self.setWindowFlags(Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        # 设置窗口拖动相关的属性
        self.drag_position = QPoint()
        # 关闭窗口美化
        icon_close = qtawesome.icon('mdi.close', color='#7b8290')
        self.close_button.setIcon(icon_close)
        self.close_button.setIconSize(QSize(30, 30))
        # 关闭窗口事件
        self.close_button.clicked.connect(self.close)
        self.cancel_button.clicked.connect(self.close)
        # 确认窗口事件
        self.confirm_button.clicked.connect(self.change_config)
        # ------定义设置类------ #
        # 文件目录
        self.curPath = os.path.abspath(os.path.dirname(__file__))
        # 项目根路径
        self.rootPath = self.curPath[:self.curPath.rindex('')]
        # 配置文件夹路径
        self.conf_folder_path = self.rootPath + '\\conf\\'
        self.setting_conf = configparser.ConfigParser()
        self.setting_conf.read(self.conf_folder_path + 'setting.conf', encoding='gbk')
        self.capture_source = self.setting_conf.get('image_config', 'capture_source')
        if self.capture_source == '0':
            self.capture_source = int(self.capture_source)
        self.tolerance = float(self.setting_conf.get('image_config', 'tolerance'))
        self.set_size = float(self.setting_conf.get('image_config', 'set_size'))
        # 输入框居中显示
        line_text = [self.camera_address_lineEdit, self.img_size_lineEdit, self.recognition_threshold_lineEdit]
        for lineEdit in line_text:
            lineEdit.setAlignment(Qt.AlignCenter)
        self.camera_address_lineEdit.setPlaceholderText(str(self.capture_source))  # 摄像头地址
        self.img_size_lineEdit.setPlaceholderText(str(self.set_size))  # 处理图像大小
        self.recognition_threshold_lineEdit.setPlaceholderText(str(self.tolerance))  # 人脸识别阈值
        self.flag_address = 0
        self.flag_size = 0
        self.flag_threshold = 0
        # ------定义设置类------ #

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drag_position = event.globalPos() - self.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event):
        if not event.buttons() & Qt.LeftButton:
            return
        self.move(event.globalPos() - self.drag_position)
        event.accept()

    def mouseReleaseEvent(self, event):
        event.accept()

    def change_config(self):
        size = self.img_size_lineEdit.text()
        address = self.camera_address_lineEdit.text()
        threshold = self.recognition_threshold_lineEdit.text()
        self.flag_address = 1
        self.flag_size = 1
        self.flag_threshold = 1
        if address != '':
            self.setting_conf.set('image_config', 'capture_source', address)
        if size != '':
            try:
                if 0 < float(size) < 1:
                    self.setting_conf.set('image_config', 'set_size', size)
                else:
                    QMessageBox.about(self, '提示', '处理图像大小-取值在0到1之间')
                    self.flag_size = 0
            except ValueError:
                QMessageBox.about(self, '提示', '处理图像大小-输入格式有问题')
                self.flag_size = 0
        if threshold != '':
            try:
                if 0 < float(threshold) < 1:
                    self.setting_conf.set('image_config', 'tolerance', threshold)
                else:
                    QMessageBox.about(self, '提示', '人脸识别阈值-取值在0到1之间')
                    self.flag_threshold = 0
            except ValueError:
                QMessageBox.about(self, '提示', '人脸识别阈值-输入格式有误')
                self.flag_threshold = 0
        self.setting_conf.write(open(self.conf_folder_path + "setting.conf", "w"))
        if self.flag_address and self.flag_size and self.flag_threshold == 1:
            QMessageBox.about(self, '提示', '配置已重置')


# 信息展示窗口
class UserShowWin(QDialog):
    def __init__(self, sid, name, password, age, sex, more, image):
        super().__init__()
        # 在 QMainWindow 实例上加载 UI
        uic.loadUi("./ui/show.ui", self)
        # 设置窗口标志以隐藏标题栏和外围边框
        self.setWindowFlags(Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        # 设置窗口拖动相关的属性
        self.drag_position = QPoint()
        # 窗口美化
        self.component_beautification()
        # 窗口事件
        self.close_button.clicked.connect(self.close)
        self.confirm_button.clicked.connect(self.close)
        # 定义
        self.sid = sid
        self.name = name
        self.password = password
        self.age = age
        self.sex = sex
        self.more = more
        self.image = image
        # 内容显示
        self.sid_lineEdit.setText(str(self.sid))
        self.name_lineEdit.setText(str(self.name))
        self.password_lineEdit.setText(str(self.password))
        self.age_lineEdit.setText(str(self.age))
        self.sex_lineEdit.setText(str(self.sex))
        self.more_lineEdit.setText(str(self.more))

        if self.image is None:
            pass
        else:
            pixmap = QPixmap(self.image)
            self.image_label.setPixmap(pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio))

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drag_position = event.globalPos() - self.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event):
        if not event.buttons() & Qt.LeftButton:
            return
        self.move(event.globalPos() - self.drag_position)
        event.accept()

    def mouseReleaseEvent(self, event):
        # 可选：实现鼠标释放事件的处理逻辑
        event.accept()

    def component_beautification(self):
        # 关闭按钮美化
        icon_close = qtawesome.icon('mdi.close', color='grey')
        self.close_button.setIcon(icon_close)
        self.close_button.setIconSize(QSize(29, 30))


if __name__ == '__main__':
    app = QApplication(sys.argv)
    SharedInformation.loginWin = WinLoginUi()
    SharedInformation.loginWin.show()
    sys.exit(app.exec_())
