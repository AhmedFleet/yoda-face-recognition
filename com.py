import cv2
import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer
import psycopg2
import os
import matplotlib.pyplot as plt

# ========== تحميل نموذج CLIP ==========
model = SentenceTransformer("clip-ViT-B-32")

# ========== تحميل كاشف الوجوه ==========
haar_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# ========== مسار صورة الاستعلام ==========
query_image_path = "testFaces/alan.png"  # ← عدّل حسب اسم صورتك
img = cv2.imread(query_image_path)

if img is None:
    print(f"[❌] الصورة غير موجودة: {query_image_path}")
    exit()

# ========== اكتشاف الوجه ==========
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = haar_cascade.detectMultiScale(gray_img, scaleFactor=1.001, minNeighbors=380, minSize=(100, 100))

if len(faces) == 0:
    print("[⚠️] لا يوجد وجه في الصورة.")
    exit()

(x, y, w, h) = faces[0]
cropped_face = img[y:y + h, x:x + w]

try:
    face_pil = Image.fromarray(cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB)).convert("RGB")
    face_pil = face_pil.resize((224, 224))
except Exception as e:
    print(f"[❌] فشل معالجة الوجه: {e}")
    exit()

# ========== حساب embedding ==========
try:
    embedding = model.encode(face_pil)
    print(f"[✅] حجم embedding: {len(embedding)}")
except Exception as e:
    print(f"[❌] فشل في حساب embedding: {e}")
    exit()

# ========== تحويل embedding إلى صيغة PostgreSQL vector ==========
vector_str = f"[{', '.join(map(str, embedding))}]"

# ========== الاتصال بقاعدة البيانات ==========
try:
    conn = psycopg2.connect(
        host="localhost",
        dbname="postgres",
        user="postgres",
        password="123",
        port=5432
    )
    cur = conn.cursor()
except Exception as e:
    print(f"[❌] خطأ في الاتصال بقاعدة البيانات: {e}")
    exit()

# ========== استعلام PostgreSQL لايجاد أقرب وجه ==========
try:
    cur.execute("""
        SELECT picture, 1 - (embedding <=> %s::vector) AS similarity
        FROM pictures
        ORDER BY embedding <=> %s::vector
        LIMIT 1
    """, (vector_str, vector_str))
    result = cur.fetchone()
except Exception as e:
    print(f"[❌] خطأ في الاستعلام: {e}")
    conn.close()
    exit()

# ========== عرض النتيجة ==========
if result:
    name, similarity = result
    percentage = similarity * 100
    print(f"[👤] closeest face: {name}")
    print(f"[📊]: {percentage:.2f}%")
else:
    print("[❌] لا توجد نتائج.")
    conn.close()
    exit()

conn.close()

# ========== عرض الصورتين جنبًا إلى جنب ==========
query_image = Image.open(query_image_path).convert("RGB")
closest_image_path = os.path.join("stored-faces", name)
closest_image = Image.open(closest_image_path).convert("RGB")

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(query_image)
axes[0].set_title("صورة الاستعلام")
axes[0].axis("off")

axes[1].imshow(closest_image)
axes[1].set_title(f"أقرب وجه: {name}\n({percentage:.2f}%)")
axes[1].axis("off")

plt.tight_layout()
plt.show()
