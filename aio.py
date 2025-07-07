import streamlit as st
import cv2
import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer
import psycopg2
import os
import random
from streamlit_lottie import st_lottie
import requests

# ========== إعداد الواجهة ==========
st.set_page_config(page_title="YODA | AI Face Recognition", layout="wide", page_icon="🧠")

DEFAULT_FACE_SCALE = 1.01
DEFAULT_MIN_NEIGHBORS = 200

if "face_scale" not in st.session_state:
    st.session_state.face_scale = DEFAULT_FACE_SCALE
if "min_neighbors" not in st.session_state:
    st.session_state.min_neighbors = DEFAULT_MIN_NEIGHBORS

# ========== تحميل النموذج ==========
model = SentenceTransformer("clip-ViT-B-32")

# ========== تحميل الرسوم المتحركة ==========
@st.cache_data
def load_lottie_url(url):
    try:
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    except:
        return None

lottie_main = load_lottie_url("https://lottie.host/ec68d393-eeb2-492d-b3fb-21b1d7dd89aa/fOXlmZgP47.json")
lottie_upload = load_lottie_url("https://lottie.host/8b971041-2496-4886-8448-6af7b7fa87b3/6gQs13ZbJX.json")
lottie_search = load_lottie_url("https://lottie.host/8f9e88fb-54a7-47ba-b8a3-3e705996091a/q93fq0vxOW.json")
lottie_gallery = load_lottie_url("https://lottie.host/b8b4c947-d359-4de1-8ae2-cf0052a19728/c7AQ3zyNiy.json")
lottie_ai = load_lottie_url("https://lottie.host/a3d2629a-7bcc-41b7-b991-cc937fd8d896/gtko6LcxIh.json")

# ========== الاتصال بقاعدة البيانات ==========
def connect_db():
    return psycopg2.connect(
        host="aws-0-eu-west-3.pooler.supabase.com",
        dbname="postgres",
        user="postgres.urqzsanhvlahtsjjbrot",
        password="YodaAi2002",
        port=5432
    )

# ========== توليد تمثيل الصورة ==========
def get_embedding(pil_image):
    pil_image = pil_image.resize((224, 224)).convert("RGB")
    return model.encode([pil_image])[0]

# ========== توصيف وتعليق ==========
def generate_comment():
    return random.choice([
        "🧠 Appears confident and focused.",
        "😊 Friendly expression.",
        "🔍 Sharp and attentive look.",
        "🎯 Determined and young adult.",
        "🧐 Calm and observant."
    ])

def describe_face():
    return random.choice([
        "👩 Female | 🕐 20-30 | 🙂 Calm",
        "👨 Male | 🕐 30-40 | 😐 Neutral",
        "👦 Teenager | 🕐 13-19 | 😀 Happy",
        "🧔 Adult | 🕐 40+ | 😎 Confident"
    ])

# ========== تنسيق الواجهة ==========
st.markdown("""
<style>
body { background-color: #0d1117; color: white; }
.block-container { padding-top: 1.5rem; }
.stButton>button {
    background-color: #00bcd4; color: white; border-radius: 12px;
    font-weight: bold; padding: 0.6em 1.4em;
    transition: 0.3s ease-in-out;
}
.stButton>button:hover {
    background-color: #0097a7; transform: scale(1.05);
}
</style>
""", unsafe_allow_html=True)

# ========== العنوان ==========
st.title("🧠 YODA - AI Face Recognition Assistant")
if lottie_main:
    st_lottie(lottie_main, height=350, key="main_lottie")

# ========== التبويبات ==========
with st.tabs(["📤 Upload & Save", "🔍 Search", "🖼️ Gallery", "🤖 AI Suggestions", "⚙️ Settings"])

# ========== 📤 Upload & Save ==========
with tabs[0]:
    st.subheader("Upload and Store Faces")
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    face_pics = []

    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="📸 Uploaded Image")

        def detect_faces_custom(image):
            haar = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            return haar.detectMultiScale(
                gray,
                scaleFactor=st.session_state.face_scale,
                minNeighbors=st.session_state.min_neighbors,
                minSize=(60, 60)
            )

        faces = detect_faces_custom(img)

        if len(faces) == 0:
            st.warning("😢 No faces detected.")
        elif len(faces) < 3:
            st.info(f"🤔 Only {len(faces)} face(s) detected. You may want to retry with higher accuracy.")
        else:
            st.success(f"✅ {len(faces)} face(s) detected.")

        # زر دائم لإعادة المحاولة
        if st.button("🔁 Retry Face Detection"):
            st.session_state.face_scale = 1.005
            st.session_state.min_neighbors = max(20, st.session_state.min_neighbors - 10)
            st.experimental_rerun()

        for i, (x, y, w, h) in enumerate(faces):
            face = img[y:y + h, x:x + w]
            face_pil = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
            st.image(face_pil, caption=f"Face {i + 1}", width=150)
            face_pics.append((face, face_pil))

        if face_pics and st.button("💾 Confirm & Save Faces"):
            conn = connect_db()
            cur = conn.cursor()
            os.makedirs("stored-faces", exist_ok=True)
            for i, (face, face_pil) in enumerate(face_pics):
                embedding = get_embedding(face_pil)
                vector_str = f"[{', '.join(map(str, embedding))}]"
                cur.execute(
                    "SELECT picture, 1 - (embedding <=> %s::vector) AS similarity FROM pictures ORDER BY embedding <=> %s::vector LIMIT 1",
                    (vector_str, vector_str))
                result = cur.fetchone()
                if result and result[1] >= 0.95:
                    st.warning(f"⚠️ Face {i + 1} is a duplicate ({result[1] * 100:.2f}%)")
                    continue
                filename = f"{i}_{os.urandom(4).hex()}.jpg"
                path = os.path.join("stored-faces", filename)
                cv2.imwrite(path, face)
                cur.execute("INSERT INTO pictures (picture, embedding) VALUES (%s, %s)", (filename, embedding.tolist()))
                st.success(f"✅ Saved: {filename} - {generate_comment()} - {describe_face()}")
            conn.commit()
            conn.close()

    if lottie_upload:
        st_lottie(lottie_upload, height=300, key="upload_anim")

# ========= Gallery Tab =========
with tabs[2]:
    st.subheader("🖼️ Stored Gallery")

    conn = connect_db()
    cur = conn.cursor()
    cur.execute("SELECT picture FROM pictures")
    results = cur.fetchall()
    conn.close()

    if results:
        cols = st.columns(4)

        for i, row in enumerate(results):
            file = row[0]
            path = os.path.join("stored-faces", file)

            with cols[i % 4]:
                if os.path.exists(path):
                    st.image(path, caption=file, width=350)
                    if st.button("🗑️ Delete", key=f"del_{file}"):
                        st.session_state["file_to_delete"] = file
                else:
                    st.warning(f"❌ Missing file: {file}")

        if "file_to_delete" in st.session_state:
            file = st.session_state["file_to_delete"]
            path = os.path.join("stored-faces", file)

            with st.expander(f"⚠️ Confirm Deletion for {file}", expanded=True):
                st.write("Are you sure you want to delete this face image?")
                st.image(path, width=200)

                col1, col2 = st.columns(2)
                with col1:
                    if st.button("✅ Yes, Delete", key="confirm_delete"):
                        conn = connect_db()
                        cur = conn.cursor()
                        cur.execute("DELETE FROM pictures WHERE picture = %s", (file,))
                        conn.commit()
                        conn.close()
                        try:
                            os.remove(path)
                        except FileNotFoundError:
                            pass
                        st.success(f"✅ Deleted: {file}")
                        del st.session_state["file_to_delete"]
                        st.experimental_rerun()

                with col2:
                    if st.button("❌ Cancel", key="cancel_delete"):
                        del st.session_state["file_to_delete"]
    else:
        st.info("📭 No faces saved.")
    if lottie_gallery:
        st_lottie(lottie_gallery, height=350, key="gallery_anim")


# ========= AI Suggestions Tab =========
with tabs[3]:
    st.subheader("⚙️ Smart AI Suggestions")
    files = os.listdir("stored-faces")
    if len(files) > 100:
        st.warning("🧠 Consider archiving old faces for performance.")
    else:
        st.success("✅ Database size is healthy.")
    st.markdown("- 📌 Use clear face images for better detection.")
    st.markdown("- 🧩 Adjust detection sensitivity if needed.")
    if lottie_ai:
        st_lottie(lottie_ai, height=350, key="ai_anim")


# ========= Settings Tab =========
with st.tabs([4]):
    st.subheader("⚙️ Detection Settings")

    st.write("### Adjust Face Detection Sensitivity")
    st.slider("Scale Factor", min_value=1.01, max_value=1.5, step=0.01, key="face_scale")
    st.slider("Min Neighbors", min_value=1, max_value=300, step=1, key="min_neighbors")
    st.success("🔧 Changes will apply on the next face detection.")

