import cv2
from PIL import Image
from torchvision import transforms
from sentence_transformers import SentenceTransformer
import psycopg2
import os
import shutil

# =============== تحميل نموذج CLIP ===============
model = SentenceTransformer("clip-ViT-B-32")
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# =============== تحميل كاشف الوجوه ===============
haar_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# =============== تحميل الصورة ===============
file_name = "two-and-a-half-men-cast-stars-characters-malibu-beach-gallery-chuck-lorre-productions-the-tannenbaum-company-warner-bros-television-cbs_1-2241006998.jpg"
img = cv2.imread(file_name)
if img is None:
    print(f"[❌] لم يتم العثور على الصورة: {file_name}")
    exit()

gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# =============== كشف الوجوه ===============
faces = haar_cascade.detectMultiScale(
    gray_img,
    scaleFactor=1.1,
    minNeighbors=20,
    minSize=(50, 50)
)
print(f"[ℹ️] عدد الوجوه المكتشفة: {len(faces)}")

# =============== حذف المجلد القديم وإنشاء جديد ===============
if os.path.exists("stored-faces"):
    shutil.rmtree("stored-faces")
os.makedirs("stored-faces", exist_ok=True)
# =============== قص الوجوه وحفظها ===============
for i, (x, y, w, h) in enumerate(faces):
    cropped_face = img[y:y + h, x:x + w]
    face_path = f"stored-faces/{i}.jpg"
    cv2.imwrite(face_path, cropped_face)
    print(f"[✔] تم حفظ الوجه: {face_path}")

# =============== الاتصال بقاعدة البيانات ===============
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
    print(f"[❌] فشل الاتصال بقاعدة البيانات: {e}")
    exit()

# =============== حذف الصور القديمة من قاعدة البيانات ===============
try:
    cur.execute("DELETE FROM pictures")
    conn.commit()
    print("[🗑] تم حذف جميع الصور القديمة من قاعدة البيانات.")
except Exception as e:
    print(f"[❌] فشل حذف الصور من قاعدة البيانات: {e}")
    conn.close()
    exit()

# =============== معالجة الصور الجديدة وحساب embeddings ===============
for picture in os.listdir("stored-faces"):
    image_path = os.path.join("stored-faces", picture)

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
        print(f"[✅] {picture} - حجم embedding: {len(embedding)}")
    except Exception as e:
        print(f"[⚠️] فشل في حساب embedding لـ {picture}: {e}")
        continue

    # إدخال إلى قاعدة البيانات
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
print("[✅] تم الانتهاء بنجاح.")
