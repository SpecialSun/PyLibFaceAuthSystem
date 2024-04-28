import numpy as np
import pymysql


def init_conn():
    conn = pymysql.connect(
        host="127.0.0.1",  # 数据库的IP地址
        user="root",  # 数据库用户名称
        password="1234",  # 数据库用户密码
        db="ai_database",  # 数据库名称
        port=3306,  # 数据库端口名称
        charset="utf8"  # 数据库的编码方式
    )
    return conn


def execute_with_bool(sql_str, args=()):
    conn = init_conn()
    cursor = conn.cursor()
    try:
        cursor.execute(sql_str, args)
        conn.commit()
        return True
    except Exception as e:
        conn.rollback()
        print(e)
        return False
    finally:
        cursor.close()


def execute_with_list(sql_str):
    conn = init_conn()
    cursor = conn.cursor()
    results = []
    try:
        cursor.execute(sql_str)
        results = cursor.fetchall()
    except Exception as e:
        conn.rollback()
        print(e)
    finally:
        cursor.close()
    return results


def search_all_msg():
    return execute_with_list("select * from user")


# 定义一个函数来从数据库中获取人脸特征
def parse_numpy_array_from_bytes(b):
    # 假设数组数据是以二进制形式存储的，我们可以直接使用 np.frombuffer 来解析
    face_encoding = np.frombuffer(b, dtype=np.float32)
    # 重塑为 (1, 512)，如果原数组形状是 (512,) 的话
    face_encoding = face_encoding.reshape(1, -1)
    return face_encoding


def get_face_encodings_from_db(cursor):
    # 执行SQL查询，选择sid、name和face_encodings字段
    cursor.execute("SELECT sid, name, face_encoding FROM user")
    # 从数据库中获取所有查询结果
    results = cursor.fetchall()
    # 初始化字典，用于存储人脸编码信息
    face_encodings = {}
    # 遍历查询结果中的每一行
    for row in results:
        # 从结果行中解包出id、name和face_encodings的字节数据
        sid, name, face_encoding_bytes = row
        # 解析字节数据，将其转换为NumPy数组（此函数需要您自己实现或提供）
        face_encoding = parse_numpy_array_from_bytes(face_encoding_bytes)
        # 在字典中添加一个条目，其中包含sid、name和转换后的人脸编码NumPy数组
        face_encodings[sid] = {'name': name, 'encoding': face_encoding}
    # 返回包含所有人脸编码信息的字典
    return face_encodings
