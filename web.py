import streamlit as st
import cv2
from PIL import Image
import numpy as np
from sentence_transformers import SentenceTransformer
import psycopg2
import tempfile
import os

# ========== تحميل نموذج CLIP ==========
model = SentenceTransformer("clip-ViT-B-32")

# ========== تحميل كاشف الوجوه ==========
haar_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# ========== واجهة Streamlit ==========
st.title("🔍 نظام التعرف على الوجوه")

uploaded_file = st.file_uploader("ارفع صورة تحتوي على وجه 👤", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="الصورة التي تم تحميلها", use_column_width=True)

    # حفظ مؤقت للمعالجة بـ OpenCV
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        image.save(tmp.name)
        img_cv = cv2.imread(tmp.name)

    gray_img = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    faces = haar_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=10)

    if len(faces) == 0:
        st.warning("❌ لم يتم اكتشاف أي وجه.")
    else:
        x, y, w, h = faces[0]
        face = img_cv[y:y+h, x:x+w]
        face_pil = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB)).resize((224, 224))

        try:
            embedding = model.encode(face_pil)
            vector_str = f"[{', '.join(map(str, embedding))}]"
        except Exception as e:
            st.error(f"[❌] فشل في حساب embedding: {e}")
            st.stop()

        # الاتصال بقاعدة البيانات PostgreSQL
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
            st.error(f"[❌] خطأ في الاتصال بقاعدة البيانات: {e}")
            st.stop()

        try:
            cur.execute("""
                SELECT picture, 1 - (embedding <=> %s::vector) AS similarity
                FROM pictures
                ORDER BY embedding <=> %s::vector
                LIMIT 1
            """, (vector_str, vector_str))
            result = cur.fetchone()
            conn.close()

            if result:
                name, similarity = result
                percent = similarity * 100
                st.success(f"👤 أقرب وجه: {name}")
                st.metric("نسبة التشابه", f"{percent:.2f}%")
                stored_face_path = os.path.join("stored-faces", name)
                if os.path.exists(stored_face_path):
                    st.image(stored_face_path, caption="أقرب وجه مطابق", use_column_width=True)
                else:
                    st.warning("تم العثور على النتيجة، لكن الصورة غير موجودة في التخزين.")
            else:
                st.warning("❌ لم يتم العثور على وجه مشابه في قاعدة البيانات.")
        except Exception as e:
            st.error(f"[❌] خطأ في الاستعلام: {e}")
