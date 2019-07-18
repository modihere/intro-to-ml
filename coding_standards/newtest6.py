import pymysql

def get_connection():
    try:
        connection = pymysql.connect(host='localhost',user='root',password='abzooba@123',db='testing')
        return connection
    except Exception as e:
        return e

connect = get_connection()
print(connect)