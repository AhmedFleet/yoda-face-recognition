import os
import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer
import psycopg2

# =============== تحميل نموذج CLIP ===============
model = SentenceTransformer("clip-ViT-B-32")

# =============== إعداد قاعدة البيانات ===============
conn = psycopg2.connect(
    host="localhost",
    dbname="postgres",
    user="postgres",
    password="123",
    port=5432
)
cur = conn.cursor()

# =============== المسار إلى مجلد الصور ===============
folder_path = "stored-faces"

# =============== تمر على الصور وتتحقق وتخزنها ===============
for picture in os.listdir(folder_path):
    image_path = os.path.join(folder_path, picture)

    # حذف الصور الفارغة
    if os.path.getsize(image_path) == 0:
        print(f"[🗑] حذف صورة فارغة: {picture}")
        os.remove(image_path)
        continue

    # فتح الصورة
    try:
        img_pil = Image.open(image_path).convert("RGB").resize((224, 224))
    except Exception as e:
        print(f"[❌] لم يمكن فتح الصورة {picture}: {e}")
        continue

    # حساب embedding
    try:
        embedding = model.encode(img_pil)
        vector_str = f"[{', '.join(map(str, embedding))}]"
    except Exception as e:
        print(f"[⚠️] فشل في حساب embedding لـ {picture}: {e}")
        continue

    # التحقق من التكرار قبل الإدخال
    try:
        cur.execute("""
            SELECT picture, 1 - (embedding <=> %s::vector) AS similarity
            FROM pictures
            ORDER BY embedding <=> %s::vector
            LIMIT 1
        """, (vector_str, vector_str))
        result = cur.fetchone()

        if result and result[1] >= 0.95:
            print(f"[⚠️] الصورة {picture} مكررة بنسبة {result[1]*100:.2f}% - تم تجاهلها")
            continue
    except Exception as e:
        print(f"[❌] خطأ أثناء فحص التكرار: {e}")
        continue

    # إدخال الصورة في قاعدة البيانات
    try:
        cur.execute("""
            INSERT INTO pictures (picture, embedding)
            VALUES (%s, %s)
            ON CONFLICT (picture) DO NOTHING
        """, (picture, embedding.tolist()))
        print(f"[↑] تم تخزين: {picture}")
    except Exception as e:
        print(f"[❌] خطأ في الإدخال لقاعدة البيانات لـ {picture}: {e}")
        continue

# =============== إنهاء الاتصال ===============
conn.commit()
conn.close()
print("[✅] تم معالجة الصور بنجاح.")
