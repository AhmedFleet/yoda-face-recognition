import psycopg2

def test_postgres_connection():
    try:
        conn = psycopg2.connect(
            host="localhost",       # أو 127.0.0.1 إذا لم تعمل
            dbname="postgres",      # غيّر الاسم إذا كانت قاعدة بياناتك مختلفة
            user="postgres",        # غيّر اسم المستخدم إذا لزم
            password="123",         # غيّر كلمة المرور حسب إعداداتك
            port=5432               # غيّر المنفذ إذا كان مختلفًا
        )
        print("[✅] تم الاتصال بقاعدة البيانات بنجاح.")
        conn.close()
    except Exception as e:
        print("[❌] فشل الاتصال بقاعدة البيانات:")
        print(e)

# تشغيل الاختبار
test_postgres_connection()