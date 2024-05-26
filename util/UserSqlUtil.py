# -*- coding: utf-8 -*-
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


def insert_data(sid, name, password, age, sex, more, face_encoding, image_path):
    return execute_with_bool(
        "insert into user(sid,name,password,age,sex,more,face_encoding,image_path) values(%s,%s,%s,%s,%s,%s,%s,%s)",
        (sid, name, password, age, sex, more, face_encoding, image_path))


def update_by_name(sid, name, password, age, sex, more, face_encoding):
    return execute_with_bool(
        "update user set sid==%s,name=%s,password=%s,age=%s,sex=%s,more=%s,face_encoding=%s where sid = %s",
        (sid, name, password, age, sex, more, face_encoding, sid))


def update_by_name_without_encoding(sid, age, sex, more):
    return execute_with_bool("update user set sid=%s,age=%s,sex=%s,more=%s where sid = %s",
                             (sid, age, sex, more, sid))


def update_by_id_without_encoding(sid, name, password, age, sex, more):
    return execute_with_bool("update user set sid=%s,name=%s,password=%s,age=%s,sex=%s,more=%s where sid = %s",
                             (sid, name, password, age, sex, more, sid))


def search_all_msg():
    return execute_with_list("select * from user")


def search_by_name(sid):
    return execute_with_list("select * from user where sid = " + sid)


def search_count_name(name):
    return execute_with_list("select count(*) from user where name = " + name)[0][0]


def delete_by_name(sid):
    return execute_with_bool("delete from user where sid = %s", sid)


def search_count_warn(sid):
    return execute_with_list("select count(*) from warn where sid = " + sid)[0][0]
