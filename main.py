import os
import sys
import cv2
import face_recognition
import configparser
import numpy as np
import torch
import pymysql
import torchvision.transforms as transforms
import threading
import pyttsx3
import openpyxl
from scipy.spatial.distance import cosine
from PIL import Image, ImageDraw, ImageFont
from facenet_pytorch import MTCNN, InceptionResnetV1
from datetime import datetime, timedelta
from time import time
from util import FaceEncodingManager
# PyQt5界面设计相关
from PyQt5.QtCore import Qt, QPoint, QSize
from PyQt5.QtGui import *
from PyQt5.QtGui import QImage
from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox, QDialog
from PyQt5 import QtWidgets, QtGui, QtCore, uic
import qtawesome
# UI界面相关
from util import UserSqlUtil
from util.share import SharedInformation
from admin_main import WinLoginUi

# 文件目录
curPath = os.path.abspath(os.path.dirname(__file__))
# 项目根路径
rootPath = curPath[:curPath.rindex('')]
# 配置文件夹路径
CONF_FOLDER_PATH = rootPath + '\\conf\\'
# 图片文件夹路径
PHOTO_FOLDER_PATH = rootPath + '\\photo\\'
# 数据文件夹路径
DATA_FOLDER_PATH = rootPath + '\\data\\'

# 登录标志
USER_LOGIN_FLAG = False
USER_LOGIN_COUNT = 0
USER_LOGIN_NAME = ""
# 临时变量
SHOT_TEMP_NAME = ""
ENCODING_TEMP = ""
is_cap_opened_flag = False
complete_registration = False


# 主窗口类
class WinMainUi(QMainWindow):
    def __init__(self, device=None):
        super().__init__()
        # 在 QMainWindow 实例上加载 UI
        uic.loadUi("./ui/win_main.ui", self)
        # 设置窗口标志以隐藏标题栏和外围边框
        self.setWindowFlags(Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        # 设置窗口拖动相关的属性
        self.drag_position = QPoint()
        # 初始化窗口状态
        self.is_maximized = False
        # -----------------------其他定义标记----------------------- #
        self.camera_start_time = None
        self.detect_start_time = None
        # 相机部分
        self.cap = cv2.VideoCapture()

        self.WIN_WIDTH = 880
        self.WIN_HEIGHT = 550
        # 人脸识别部分
        self.isFaceRecognition_flag = False
        self.all_list = []
        self.numpy_features = []
        self.process_this_frame = True
        self.matched_name = ''
        self.set_name = None
        self.set_names = None
        self.unknown_names = []
        self.pixmap_with_text = None
        self.isFaceRecognition = False
        self.user_id = None
        self.min_face_size = 100
        self.recent_names = {}
        self.show_image = None
        self.set_name = None
        self.set_names = None
        self.face_features = []
        # -----------------------其他定义标记----------------------- #
        # 设置时间窗口为5秒
        self.time_window = timedelta(seconds=60)
        # 表格部分
        self.excel_file_path_1 = './data/login_records.xlsx'
        self.excel_file_path_2 = './data/successful_identification_records.xlsx'
        self.sheet_name = 'Sheet1'  # Excel中工作表的名称
        self.sheet_login = 'Sheet1'
        self.isCapOpened_flag = False
        # 如果device没有被指定（即为None），则检查是否有可用的GPU
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        # 初始化MTCNN模型，用于人脸检测
        self.mtcnn = MTCNN(keep_all=True, device=self.device)
        # 初始化InceptionResnetV1模型，用于特征提取
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)

        # ------------------部署数据库------------------ #
        # 读取Mysql数据库配置文件
        mysql_path = './conf/dataAiBase.conf'
        config = configparser.ConfigParser()
        config.read(mysql_path)  # 读取配置文件，假设它与脚本在同一目录下
        # 从配置中获取数据库连接信息
        db_host = config['mysql']['host']
        db_user = config['mysql']['user']
        db_password = config['mysql']['password']
        db_database = config['mysql']['database']
        db = pymysql.connect(host=db_host, user=db_user, password=db_password, database=db_database)
        self.cursor = db.cursor()
        # 文件配置
        # 文件目录
        cur_path = os.path.abspath(os.path.dirname(__file__))
        # 项目根路径
        root_path = cur_path[:cur_path.rindex('')]
        # 配置文件夹路径
        self.conf_folder_path = root_path + '\\conf\\'
        # 图片文件夹路径
        self.photo_folder_path = root_path + '\\photo\\'
        # 读取人脸识别配置文件获取参数
        conf_image_config = configparser.ConfigParser()
        conf_image_config.read(self.conf_folder_path + 'setting.conf', encoding='gbk')
        capture_source = conf_image_config.get('image_config', 'capture_source')
        if capture_source == '0':
            capture_source = int(capture_source)
        self.source = capture_source

        # ------------------部署数据库------------------ #
        self.component_beautification()
        # 窗口控制
        self.close_button.clicked.connect(self.close_window)
        self.minimize_button.clicked.connect(self.minimize_window)
        # 管理员按钮
        self.admin_button.clicked.connect(self.admin_button_clicked)
        # 打开摄像头按钮
        self.camera_button.clicked.connect(self.open_camera)
        # 人脸识别按钮
        self.recognition_button.clicked.connect(self.recognize_face_judge)
        # 关于按钮
        self.about_button.clicked.connect(self.about_software)
        # 用户按钮
        self.user_button.clicked.connect(self.user_win_open)
        # 注册按钮
        self.user_register_button.clicked.connect(self.user_register)

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
        icon_close = qtawesome.icon('mdi.close', color='#7b8290')
        self.close_button.setIcon(icon_close)
        self.close_button.setIconSize(QSize(30, 30))
        # 最小化按钮美化
        icon_minimize = qtawesome.icon('mdi.minus', color='grey')
        self.minimize_button.setIcon(icon_minimize)
        self.minimize_button.setIconSize(QSize(30, 30))
        # 其他按钮
        icon_other = qtawesome.icon('mdi.window-restore', color='grey')
        self.other_button.setIcon(icon_other)
        self.other_button.setIconSize(QSize(30, 30))
        # 其他组件美化
        self.user_button.setIcon(qtawesome.icon('fa5s.user-alt', color='#787f8d'))
        self.admin_button.setIcon(qtawesome.icon('ri.admin-fill', color='#787f8d'))
        self.camera_button.setIcon(qtawesome.icon('fa5s.camera', color='#787f8d'))
        self.recognition_button.setIcon(qtawesome.icon('mdi.face-recognition', color='#787f8d'))
        self.about_button.setIcon(qtawesome.icon('fa.meetup', color='#787f8d'))
        self.user_register_button.setIcon(qtawesome.icon('fa5s.user-alt', color='#787f8d'))

    def close_window(self):
        # 弹出提示框
        reply = QMessageBox.question(self, '关闭窗口',
                                     "你确定退出吗?", QMessageBox.Yes |
                                     QMessageBox.No, QMessageBox.No)
        # 检查用户的响应
        if reply == QMessageBox.Yes:
            exit()

    def minimize_window(self):
        self.showMinimized()

    def camera_clicked_min(self):
        self.camera_label.setPixmap(QPixmap(""))
        self.camera_label.setText('    \n\n基于人脸识别的图书馆门禁系统')
        self.camera_label.setAlignment(Qt.AlignCenter)
        # 设置文本颜色
        self.camera_label.setStyleSheet('''QLabel{
                color: #787f8d;
                font-size:60px;
                font-weight:1000;
                border-top-right-radius:5px;
                border-bottom-right-radius:5px;}''')

    @staticmethod
    def admin_button_clicked():
        SharedInformation.loginWin = WinLoginUi()
        SharedInformation.loginWin.show()

    # ----------------- 打开摄像头 ------------------ #
    # 打开摄像头判断器
    def open_camera(self):
        global is_cap_opened_flag
        if not self.cap.isOpened():
            self.camera_start_time = time()
            self.cap.open(self.source)
            print(f"相机 初始化时间: {time() - self.camera_start_time:.2f}s")
            try:

                is_cap_opened_flag = True
                self.clear_information()
                self.show_camera()
            except ValueError:
                QMessageBox.about(self, '警告', '相机不能正常被打开')
        else:
            # 关闭摄像头，释放cap
            is_cap_opened_flag = False
            self.camera_button.setText(u'打开相机')
            self.camera_clicked_min()
            self.cap.release()
            cv2.destroyAllWindows()
            QMessageBox.about(self, '提示', '相机已经关闭')
            self.clear_information()

    # 展示摄像头画面
    def show_camera(self):
        self.camera_button.setText(u'关闭相机')
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            QApplication.processEvents()
            show = cv2.resize(frame, (self.WIN_WIDTH, self.WIN_HEIGHT))
            show = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)
            self.show_image = QImage(show.data, show.shape[1], show.shape[0], QImage.Format_RGB888)
            self.camera_label.setPixmap(QPixmap.fromImage(self.show_image))
        self.camera_label.setPixmap(QPixmap(""))

    # ----------------- 用户注册 ------------------ #
    # 用户注册

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
        return frame, filtered_boxes  # 返回修改后的frame和过滤后的人脸框列表

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
                # 使用预训练的模型提取特征
                with torch.no_grad():
                    feature = self.resnet(face_tensor).cpu()
                    face_features.append(feature)
        return face_features

    def _prepare_face_tensor(self, face):
        """
        将裁剪出的人脸图像转换为模型所需的Tensor格式。
        """
        if face.size == 0:
            print("Warning: Empty face image, skipping tensor preparation.")
            return None  # 或者返回其他适当的值，比如一个全零的Tensor

        face = cv2.resize(face, (160, 160))  # 调整图像大小以适应模型输入
        face = face.astype(np.float32) / 255.0  # 归一化图像
        face = face.transpose((2, 0, 1))  # 转换图像维度顺序
        face_tensor = torch.from_numpy(face).unsqueeze(0).to(self.device)  # 转换为Tensor并添加到设备
        return face_tensor

    # 将特征向量列表（PyTorch张量）转换为NumPy数组列表，以便于存储到数据库中
    def convert_features_to_array(self, face_features):
        for feature in face_features:
            # 将PyTorch张量转换为NumPy数组，并添加到列表中
            numpy_feature = feature.squeeze().cpu().numpy()
            self.numpy_features.append(numpy_feature)
        return self.numpy_features

    def match_face_with_database(self, numpy_feature):
        QApplication.processEvents()
        # 使用保存的游标来调用 get_face_encodings_from_db
        face_encodings = FaceEncodingManager.get_face_encodings_from_db(self.cursor)
        best_match_name = None
        min_distance = float('inf')
        for id, data in face_encodings.items():
            QApplication.processEvents()
            distance = cosine(numpy_feature, data['encoding'])
            if distance < min_distance:
                min_distance = distance
                best_match_name = data['name']
        if min_distance < 0.5:  # 假设0.5是我们的相似度阈值
            return best_match_name
        else:
            return "未匹配到人脸"

    # ----------------- 人脸识别 ------------------ #
    # 人脸识别判断器
    def recognize_face_judge(self):
        if not self.cap.isOpened():
            QMessageBox.information(self, "提示", self.tr(u"请先打开摄像头"))
        else:
            # 点击人脸识别时，人脸识别是关闭的
            if not self.isFaceRecognition_flag:
                self.isFaceRecognition_flag = True
                self.recognition_button.setText(u'关闭人脸识别')
                self.recognize_continuous_face()
            # 点击人脸识别时，人脸识别已经开启
            elif self.isFaceRecognition_flag:
                self.isFaceRecognition_flag = False
                self.recognition_button.setText(u'人脸识别')
                self.clear_information()
                self.show_camera()

    # 持续人脸识别
    def recognize_continuous_face(self):
        self.read_person_msg()
        last_matched_name = None  # 存储上一次匹配到的名字
        last_update_time = time()  # 存储上一次更新的时间
        update_interval = 3  # 设置更新的时间阈值，例如0.5秒
        while True:
            ret, frame = self.cap.read()
            QtWidgets.QApplication.processEvents()
            if not ret:
                print("Failed to capture frame")
                break
            frame_with_boxes, boxes = self.detect_faces(frame.copy())
            # 转换图像格式以便在 QLabel 中显示
            rgb_image = cv2.cvtColor(frame_with_boxes, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
            pixmap = QtGui.QPixmap.fromImage(qt_image)
            pixmap = pixmap.scaled(self.camera_label.size(), QtCore.Qt.KeepAspectRatio)
            self.camera_label.setPixmap(pixmap)
            if boxes:
                for box in boxes:
                    x1, y1, x2, y2 = box.astype("int")
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    face = frame[y1:y2, x1:x2]
                    face_tensor = self._prepare_face_tensor(face)
                    if face_tensor is not None:
                        with torch.no_grad():
                            feature = self.resnet(face_tensor).cpu()
                        self.matched_name = self.match_face_with_database(feature)
                        current_matched_name = self.matched_name
                        # 实现去重和防抖
                        if (current_matched_name != last_matched_name) or (
                                time() - last_update_time > update_interval):
                            last_matched_name = current_matched_name
                            last_update_time = time()

                            # 使用 PIL 在图像上绘制中文文本
                            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                            draw = ImageDraw.Draw(pil_image)
                            font_path = 'data/AlimamaDaoLiTi.ttf'  # 替换为你的中文字体文件路径
                            font_size = 26
                            font = ImageFont.truetype(font_path, font_size)
                            # 注意：textbox的第一个参数是文本的起始位置，这里我们假设从(0, 0)开始
                            bbox = draw.textbbox((0, 0), self.matched_name, font)

                            # 计算文本的宽度和高度
                            # text_width = bbox[2] - bbox[0]  # 右边界减去左边界
                            text_height = bbox[3] - bbox[1]  # 下边界减去上边界

                            text_x = x1 + 5  # 文本开始的 x 坐标，根据需要调整
                            text_y = y1 - text_height - 5  # 文本开始的 y 坐标，确保在框内且不会遮挡脸部
                            draw.text((text_x, text_y), self.matched_name, font=font, fill=(255, 255, 255))

                            # 将 PIL 图像转回 OpenCV 格式并显示在 QLabel 上
                            frame_with_text = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
                            rgb_image_with_text = cv2.cvtColor(frame_with_text, cv2.COLOR_BGR2RGB)
                            qt_image_with_text = (
                                QtGui.QImage(rgb_image_with_text.data, w, h, bytes_per_line,
                                             QtGui.QImage.Format_RGB888))
                            pixmap_with_text = QtGui.QPixmap.fromImage(qt_image_with_text)
                            self.pixmap_with_text = pixmap_with_text.scaled(self.camera_label.size(),
                                                                            QtCore.Qt.KeepAspectRatio)

                            if self.matched_name != '' and self.matched_name != '未匹配到人脸':
                                self.isFaceRecognition = True
                                self.set_name = set(self.unknown_names)
                                self.set_names = tuple(self.set_name)
                                self.unknown_names.append(self.matched_name)
                                self.show_results()
                            elif self.matched_name == '未匹配到人脸':
                                self.isFaceRecognition = False
                                self.clear_information()
                                self.speak_text_in_thread("认证失败，请重试！")
                            else:
                                self.clear_information()
                                self.isFaceRecognition = False
                                print("2")
                    else:
                        print("Warning: Unable to prepare face tensor, skipping face matching.")
            else:
                self.clear_information()
                self.isFaceRecognition = False
            # 假设我们交替处理帧以节省计算资源
            self.process_this_frame = not self.process_this_frame
        self.cap.release()
        cv2.destroyAllWindows()

    def speak_text_in_thread(self, text):
        # 使用TextToSpeech类中的静态方法来启动线程，并处理线程状态
        if TextToSpeech.thread_running:
            return
        thread = threading.Thread(target=self.speak_text, args=(text,))
        thread.start()

    def user_register(self):
        if not self.cap.isOpened():
            QMessageBox.information(self, "提示", self.tr("请先打开摄像头!"))
        else:
            ret, frame = self.cap.read()
            processed_frame, boxes = self.detect_faces(frame)  # Keep the boxes for feature extraction
            frame_location = face_recognition.face_locations(frame)
            if boxes is not None:
                face_features = self.extract_face_features(frame)  # Extract features
                self.numpy_features = self.convert_features_to_array(face_features)  # 转换特征为NumPy数组
                if len(frame_location) != 0 and len(self.numpy_features) != 0:
                    QMessageBox.information(self, "提示", self.tr("拍照成功!"))
                    # 显示帧
                    cv2.imshow('Face Recognition', processed_frame)
                    global PHOTO_FOLDER_PATH
                    global SHOT_TEMP_NAME
                    SHOT_TEMP_NAME = datetime.now().strftime("%Y%m%d%H%M%S")
                    self.show_image.save(PHOTO_FOLDER_PATH + SHOT_TEMP_NAME + ".jpg")
                    SharedInformation.userRegisterWin = UserRegisterWindow()
                    SharedInformation.userRegisterWin.exec_()
                else:
                    QMessageBox.information(self, "错误", self.tr("没有人脸检测到人脸，请重新拍照"))
            else:
                QMessageBox.information(self, "错误", self.tr("没有人脸检测到人脸，请重新拍照"))

    @staticmethod
    def speak_text(text):
        # 这里调用TextToSpeech的静态方法，但不需要关心线程状态，因为那已经在静态方法中处理了
        if '成功' in text:
            TextToSpeech.speak_in_thread_success(text)
        else:
            TextToSpeech.speak_in_thread_failure(text)

    # 读取用户信息
    def read_person_msg(self):
        all_msg = UserSqlUtil.search_all_msg()
        for tup in all_msg:
            per_list = []
            for i in tup:
                per_list.append(i)
            self.all_list.append(per_list)

    # 展示人脸识别结果
    def show_results(self):
        if self.isFaceRecognition:
            msg_label = {0: self.msg_label_a,
                         1: self.msg_label_b,
                         2: self.msg_label_c}
            # 最多放置3个人的信息
            print(self.matched_name)
            if self.set_names is None:
                print("self.set_names is None, cannot get its length.")
                # 这里可以设置show_person的默认值，或者抛出一个异常，或者进行其他错误处理
                show_person = 0  # 假设设置为0作为默认值
            elif len(self.set_names) > 1:
                show_person = 1
            else:
                show_person = len(self.set_names)
            if show_person != 0:
                for show_index in range(show_person):
                    name = self.matched_name
                    try:
                        per_label = msg_label[show_index]
                        index = self.search_person_index(self.all_list, name)
                        if index != -1:
                            infor_str = '姓名: ' + name + '                ' + \
                                        ' 年龄: ' + self.all_list[index][3] + '                 ' + \
                                        ' 性别: ' + self.all_list[index][4] + '                 ' + \
                                        ' 更多: ' + self.all_list[index][5]

                            self.user_id = self.all_list[index][0]
                            per_label.setText(infor_str)
                            self.camera_label.setPixmap(self.pixmap_with_text)
                            per_label.setStyleSheet("color:black;font-size:22px;font-family:Microsoft YaHei;")
                            per_label.setWordWrap(True)
                            self.speak_text_in_thread("认证成功")
                    except ValueError:
                        QMessageBox.about(self, '警告', '请检查' + name + '的信息')
                    if self.user_id is not None:
                        self.save_excel()
            if show_person != 3:
                for empty in range(1)[show_person:]:
                    per_label = msg_label[empty]
                    per_label.setText("")

    def write_to_excel(self, sheet):
        # 获取当前时间并格式化
        str_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        device = '摄像头'
        operation = '识别成功'

        # 假设self.set_names是一个元组，我们取第一个元素作为姓名
        if isinstance(self.set_names, tuple) and len(self.set_names) > 0:
            name = self.set_names[0]
        else:
            # 如果self.set_names不是元组或者为空，直接将其作为姓名
            name = self.set_names
        current_time = datetime.now()
        if name in self.recent_names and \
                current_time - self.recent_names[name] <= self.time_window:
            # 如果name在时间窗口内已经存在，则不写入
            return
        sheet.append([str_time, device, name, operation, self.user_id])
        # 更新最近存入的姓名和时间
        self.recent_names[name] = current_time

    # 登录登出记录
    @staticmethod
    def write_to_excel_login(sheet, name, identification):
        str_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        device = '摄像头'
        sheet.append([str_time, device, name, identification])

    def save_excel(self):
        # 打开Excel文件和工作表
        workbook = openpyxl.load_workbook(self.excel_file_path_2)
        sheet = workbook[self.sheet_name]
        # 调用函数写入数据
        self.write_to_excel(sheet)

        # 保存Excel文件
        workbook.save(self.excel_file_path_2)
        workbook.close()
        # 关闭主窗口

    # 登录登出保存
    def save_excel_login(self, name, identification):
        # 打开Excel文件和工作表
        workbook = openpyxl.load_workbook(self.excel_file_path_1)
        sheet = workbook[self.sheet_login]
        # 调用函数写入数据
        self.write_to_excel_login(sheet, name, identification)

        # 保存Excel文件
        workbook.save(self.excel_file_path_1)
        workbook.close()
        # 关闭主窗口

    # 查找指定用户下标
    @staticmethod
    def search_person_index(persons_infor, name):
        for i in range(len(persons_infor)):
            if persons_infor[i][1] == name:
                return i
        return -1

    # 清除人脸识别结果信息
    def clear_information(self):
        self.msg_label_a.setPixmap(QPixmap(""))
        self.msg_label_a.setText("")
        self.msg_label_b.setPixmap(QPixmap(""))
        self.msg_label_b.setText("")
        self.msg_label_c.setPixmap(QPixmap(""))
        self.msg_label_c.setText("")

    @staticmethod
    def about_software():
        SharedInformation.aboutWin = AboutWindow()
        SharedInformation.aboutWin.exec_()

    @staticmethod
    def user_win_open():
        SharedInformation.userWin = UserWindow()
        SharedInformation.userWin.show()


# 关于软件窗口
class AboutWindow(QDialog):
    def __init__(self):
        # 初始化 QMainWindow
        super().__init__()
        # 在 QMainWindow 实例上加载 UI
        uic.loadUi("./ui/about_software.ui", self)
        # 设置窗口标志以隐藏标题栏和外围边框
        self.setWindowFlags(Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        # 设置窗口拖动相关的属性
        self.drag_position = QPoint()
        self.confirm_button.clicked.connect(self.about)
        # 关闭窗口美化
        icon_close = qtawesome.icon('mdi.close', color='grey')
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


# 用户窗口
class UserWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        # 在 QMainWindow 实例上加载 UI
        uic.loadUi("./ui/user_login.ui", self)
        # 设置窗口标志以隐藏标题栏和外围边框
        self.setWindowFlags(Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        # 设置窗口拖动相关的属性
        self.drag_position = QPoint()
        # 窗口美化
        self.component_beautification()
        # 登录按钮按下
        self.login_button.clicked.connect(self.login_window)
        # 窗口事件
        # 最小化和关闭按钮
        self.close_button.clicked.connect(self.close_window)
        self.minimize_button.clicked.connect(self.minimize_window)
        # 退出按钮
        self.quit_button.clicked.connect(self.close_window)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key_Return or event.key() == Qt.Key_Enter:
            self.on_enter_pressed()
        else:
            pass

    def on_enter_pressed(self):
        # 在这里处理 Enter 键被按下的逻辑
        self.login_window()

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
        # 最小化按钮美化
        icon_minimize = qtawesome.icon('mdi.minus', color='grey')
        self.minimize_button.setIcon(icon_minimize)
        self.minimize_button.setIconSize(QSize(29, 30))

    def login_window(self):
        input_sid = self.user_lineEdit.text()
        input_password = self.password_lineEdit.text()
        if input_sid == "":
            QMessageBox.about(self, '提示', '姓名不能为空')
        elif input_password == "":
            QMessageBox.about(self, '提示', '密码不能为空')
        else:
            row = UserSqlUtil.search_by_name("\"" + input_sid + "\"")
            if row:
                result = row[0]
                password = result[2]
                if input_password != password:
                    QMessageBox.about(self, '提示', '密码输入错误')
                else:
                    global ENCODING_TEMP
                    global USER_LOGIN_NAME

                    ENCODING_TEMP = result[5]
                    USER_LOGIN_NAME = input_sid
                    login_successful = '登录成功'
                    WinMainUi().save_excel_login(USER_LOGIN_NAME, login_successful)
                    self.hide()
                    SharedInformation.userRegisterWin = UserModifyWin()
                    SharedInformation.userRegisterWin.exec_()

            else:
                QMessageBox.about(self, '提示', '该用户不存在')

    def close_window(self):

        self.close()

    def minimize_window(self):
        self.showMinimized()


# 注册窗口
class UserRegisterWindow(QDialog):
    def __init__(self, device=None):
        super().__init__()
        # 在 QMainWindow 实例上加载 UI
        uic.loadUi("./ui/user_register.ui", self)
        # 设置窗口标志以隐藏标题栏和外围边框
        self.setWindowFlags(Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        # 设置窗口拖动相关的属性
        self.drag_position = QPoint()
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        self.mtcnn = MTCNN(keep_all=True, device=self.device)
        # 初始化InceptionResnetV1模型，用于特征提取
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        self.flag = 0
        # 参数设置
        self.face_transformer = transforms.Compose([
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        # 窗口美化
        self.component_beautification()
        # 登录按钮按下
        self.confirm_button.clicked.connect(self.fill_information)
        # 窗口事件
        # 最小化和关闭按钮
        self.close_button.clicked.connect(self.exit_window)
        # 退出按钮
        self.cancel_button.clicked.connect(self.close_window)
        # 定义
        self.flag = 0

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key_Return or event.key() == Qt.Key_Enter:
            self.on_enter_pressed()
        else:
            pass

    def on_enter_pressed(self):
        # 在这里处理 Enter 键被按下的逻辑
        self.login_window()

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

    # 填写信息
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
                    global PHOTO_FOLDER_PATH
                    global SHOT_TEMP_NAME
                    global complete_registration
                    if self.flag == 0:
                        # 计算脸部数据并保存到数据库PHOTO_FOLDER_PATH + SHOT_TEMP_NAME + ".jpg"
                        QApplication.processEvents()
                        register_encoding = self.analyse_encoding(SHOT_TEMP_NAME)
                        image_path = PHOTO_FOLDER_PATH + SHOT_TEMP_NAME + ".jpg"
                        if self.save_database(sid, name, password, age, sex, more_infor, register_encoding, image_path):
                            QMessageBox.about(self, '提示', '完成注册')
                            complete_registration = True
                        else:
                            QMessageBox.about(self, '提示', '注册失败')
                            self.delete_shot()
                    elif self.flag == 1:
                        QMessageBox.about(self, '提示', '注册失败')
                else:
                    QMessageBox.about(self, '提示', '请输入密码')
            else:
                QMessageBox.about(self, '提示', '请输入学号')
        else:
            QMessageBox.about(self, '提示', '用户' + sid + '已经注册过')

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

    # 保存注册信息
    @staticmethod
    def save_database(sid, name, password, age, sex, more, face_encoding, image_path):
        return UserSqlUtil.insert_data(sid, name, password, age, sex, more, face_encoding, image_path)

    @staticmethod
    def save_image_path(sid, image_path):
        return UserSqlUtil

    # 判断姓名是否冲突
    @staticmethod
    def judge_name_conflict(sid):
        count = UserSqlUtil.search_count_name("\"" + sid + "\"")
        if count != 0:
            return False
        else:
            return True

    # 分析截图

    def analyse_encoding(self, name):
        global PHOTO_FOLDER_PATH
        photo_path = PHOTO_FOLDER_PATH + name + ".jpg"
        face_features_numpy = self.numpy_face_features(photo_path)
        face_encodings_binary = face_features_numpy.tobytes()
        return face_encodings_binary

    # 删除截图
    @staticmethod
    def delete_shot():
        global PHOTO_FOLDER_PATH
        global SHOT_TEMP_NAME
        delete_shot_path = PHOTO_FOLDER_PATH + SHOT_TEMP_NAME + ".jpg"
        os.remove(delete_shot_path)
        SHOT_TEMP_NAME = ""

    # 关闭窗口
    def close_window(self):
        global complete_registration
        if complete_registration:
            cv2.destroyWindow('Face Recognition')
            self.close()
        else:
            complete_registration = False
            self.delete_shot()
            cv2.destroyWindow('Face Recognition')
            self.close()

    def exit_window(self):
        global complete_registration
        if complete_registration:
            cv2.destroyWindow('Face Recognition')
            self.close()
        else:
            complete_registration = False
            self.delete_shot()
            cv2.destroyWindow('Face Recognition')
            self.close()


# 修改窗口
class UserModifyWin(QDialog):
    def __init__(self):
        super().__init__()
        # 在 QMainWindow 实例上加载 UI
        uic.loadUi("./ui/user_modify.ui", self)
        # 设置窗口标志以隐藏标题栏和外围边框
        self.setWindowFlags(Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        # 设置窗口拖动相关的属性
        self.drag_position = QPoint()
        # 窗口美化
        self.component_beautification()
        # 窗口事件
        self.close_button.clicked.connect(self.close)
        self.cancel_button.clicked.connect(self.close)
        self.back_login_button.clicked.connect(self.back_user_login)
        self.sid = ""
        self.name = ""
        self.age = ""
        self.sex = ""
        self.more = ""
        self.isSearch_flag = 0
        self.hasModify_flag = 0
        self.sql = UserSqlUtil

        self.close_button.clicked.connect(self.close_window)
        self.search_title.clicked.connect(self.search_infor)
        self.cancel_button.clicked.connect(self.close_window)
        self.confirm_button.clicked.connect(self.modify_infor)

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

    def back_user_login(self):
        self.close()
        SharedInformation.userWin = UserWindow()
        SharedInformation.userWin.show()

    def search_infor(self):
        sid = self.name_lineEdit.text()
        row = self.sql.search_by_name("\"" + sid + "\"")
        if row:
            result = row[0]
            self.age = result[3]
            self.sex = result[4]
            self.more = result[5]
            self.age_lineEdit.setPlaceholderText(self.age)
            self.sex_lineEdit.setPlaceholderText(self.sex)
            self.more_lineEdit.setPlaceholderText(self.more)
            self.isSearch_flag = 1
        else:
            QMessageBox.about(self, '提示', '找不到' + sid + '的信息')

    def modify_infor(self):
        if self.isSearch_flag == 1:
            sid = self.name_lineEdit.text()
            age = self.age_lineEdit.text()
            sex = self.sex_lineEdit.text()
            more = self.more_lineEdit.text()
            if age != '' and str.isdigit(age) is True:
                self.hasModify_flag = 1
                print('')
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
                self.sql.update_by_name_without_encoding(sid, age, sex, more)
                QMessageBox.about(self, '更新', sid + '的部分信息已更改')
            else:
                QMessageBox.about(self, '提示', '请先修改信息，否则点击退出')
        else:
            QMessageBox.about(self, '提示', '请先查找用户')

    def close_window(self):
        self.isSearch_flag = 0
        self.name_lineEdit.setPlaceholderText('输入待修改者学号，点击查找')
        self.age_lineEdit.setPlaceholderText('********')
        self.sex_lineEdit.setPlaceholderText('********')
        self.more_lineEdit.setPlaceholderText('********')
        self.close()


# 语音多线程播报
class TextToSpeech:
    # 线程状态标志，初始化为False表示没有线程在运行
    thread_running = False
    lock = threading.Lock()  # 用于同步访问thread_running标志的锁

    @staticmethod
    def speak_in_thread_success(text):
        with TextToSpeech.lock:
            # 检查是否有线程正在运行
            if TextToSpeech.thread_running:
                return  # 如果有，则直接返回，不启动新线程
            TextToSpeech.thread_running = True  # 设置线程正在运行标志
        try:
            engine_success = pyttsx3.init()
            engine_success.say(text)
            engine_success.runAndWait()
        finally:
            with TextToSpeech.lock:
                TextToSpeech.thread_running = False  # 无论线程如何结束，都设置线程未运行标志

    @staticmethod
    def speak_in_thread_failure(text):
        with TextToSpeech.lock:
            # 检查是否有线程正在运行
            if TextToSpeech.thread_running:
                return  # 如果有，则直接返回，不启动新线程
            TextToSpeech.thread_running = True  # 设置线程正在运行标志

        try:
            engine_failure = pyttsx3.init()
            engine_failure.say(text)
            engine_failure.runAndWait()
        finally:
            with TextToSpeech.lock:
                TextToSpeech.thread_running = False  # 无论线程如何结束，都设置线程未运行标志


if __name__ == '__main__':
    ui_start_time = time()
    app = QApplication(sys.argv)
    win = WinMainUi()
    win.show()
    sys.exit(app.exec_())
