import pymysql


class AdminConnect:

    def __init__(self):
        self.conn = pymysql.connect(
            host="127.0.0.1",
            user="root",
            password="1234",
            database="ai_database",
            port=3306,
            charset="utf8"
        )

    def execute_with_bool(self, sql_str, args=()):
        with self.conn.cursor() as cursor:
            try:
                cursor.execute(sql_str, args)
                self.conn.commit()
                return True
            except Exception as e:
                self.conn.rollback()
                print(e)
                return False

    def execute_with_list(self, sql_str, args=()):
        with self.conn.cursor() as cursor:
            try:
                cursor.execute(sql_str, args)
                return cursor.fetchall()
            except Exception as e:
                self.conn.rollback()
                print(e)
                return []

    def search_by_id(self, admin_id):
        sql_str = "SELECT * FROM admin WHERE name = %s"
        return self.execute_with_list(sql_str, (admin_id,))

    def delete_by_id(self, admin_id):
        sql_str = "DELETE FROM admin WHERE name = %s"
        return self.execute_with_bool(sql_str, (admin_id,))
