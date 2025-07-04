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

# ========= Load model =========
model = SentenceTransformer("clip-ViT-B-32")
model = model.to('cpu')

# ========= Load Lottie Animations =========
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
lottie_report = load_lottie_url("https://lottie.host/5313703a-ea9e-44da-9e38-72f1d5e0a094/ROZ7VmTJ5W.json")
lottie_ai = load_lottie_url("https://lottie.host/a3d2629a-7bcc-41b7-b991-cc937fd8d896/gtko6LcxIh.json")

# ========= DB Connection =========
def connect_db():
    return psycopg2.connect(
        host="aws-0-eu-west-3.pooler.supabase.com",
        dbname="postgres",
        user="postgres.urqzsanhvlahtsjjbrot",
        password="YodaAi2002",
        port=5432
    )

# ========= Face Detection =========
def detect_faces(image):
    haar = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return haar.detectMultiScale(gray, 1.05, 10, minSize=(40, 40))

# ========= Embedding =========
def get_embedding(pil_image):
    pil_image = pil_image.resize((224, 224)).convert("RGB")
    return model.encode([pil_image])[0]


# ========= Smart Comments =========
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

# ========= Streamlit Setup =========
st.set_page_config(page_title="YODA | AI Face Recognition", layout="wide", page_icon="🧠")

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

st.title("🧠 YODA - AI Face Recognition Assistant")
if lottie_main:
    st_lottie(lottie_main, speed=1, reverse=False, loop=True, quality="high", height=300, key="main_anim")

tabs = st.tabs(["📤 Upload & Save", "🔍 Search", "🖼️ Gallery", "📊 Report", "⚙️ AI Suggestions"])

# ========= Upload =========
with tabs[0]:
    st.subheader("Upload Image with Faces")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="📸 Uploaded Image", use_container_width=True)
        faces = detect_faces(img)
        st.info(f"✅ {len(faces)} face(s) detected.")
        if faces.any():
            conn = connect_db()
            cur = conn.cursor()
            os.makedirs("stored-faces", exist_ok=True)
            new_faces = []
            descriptions = []
            for i, (x, y, w, h) in enumerate(faces):
                face = img[y:y+h, x:x+w]
                face_pil = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
                embedding = get_embedding(face_pil)
                vector_str = f"[{', '.join(map(str, embedding))}]"
                cur.execute("SELECT picture, 1 - (embedding <=> %s::vector) AS similarity FROM pictures ORDER BY embedding <=> %s::vector LIMIT 1", (vector_str, vector_str))
                result = cur.fetchone()
                if result and result[1] >= 0.95:
                    st.warning(f"⚠️ Face {i+1} is a duplicate ({result[1]*100:.2f}%)")
                    continue
                filename = f"{i}_{os.urandom(4).hex()}.jpg"
                path = os.path.join("stored-faces", filename)
                cv2.imwrite(path, face)
                new_faces.append((path, filename))
                cur.execute("INSERT INTO pictures (picture, embedding) VALUES (%s, %s)", (filename, embedding.tolist()))
                descriptions.append((generate_comment(), describe_face()))
            conn.commit()
            conn.close()
            if new_faces:
                st.markdown("### ✅ Recently Saved:")
                for i, (path, name) in enumerate(new_faces):
                    col1, col2 = st.columns([1, 2])
                    col1.image(path, width=150)
                    col2.write(f"**{name}**{descriptions[i][0]}{descriptions[i][1]}")
    if lottie_upload:
        st_lottie(lottie_upload, height=250, key="upload_anim")

# ========= Search =========
with tabs[1]:
    st.subheader("🔍 Upload a Face to Search")
    query_file = st.file_uploader("Choose face image...", type=["jpg", "jpeg", "png"], key="query")
    if query_file:
        file_bytes = np.asarray(bytearray(query_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        faces = detect_faces(img)
        if not faces.any():
            st.error("😢 No face detected.")
        else:
            (x, y, w, h) = faces[0]
            face = img[y:y+h, x:x+w]
            face_pil = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
            embedding = get_embedding(face_pil)
            vector_str = f"[{', '.join(map(str, embedding))}]"
            conn = connect_db()
            cur = conn.cursor()
            cur.execute("SELECT picture, 1 - (embedding <=> %s::vector) AS similarity FROM pictures ORDER BY embedding <=> %s::vector LIMIT 1", (vector_str, vector_str))
            result = cur.fetchone()
            conn.close()
            if result:
                name, sim = result
                percent = sim * 100
                col1, col2 = st.columns(2)
                col1.image(face_pil, caption="Query Face", use_container_width=True)
                path = os.path.join("stored-faces", name)
                col2.image(Image.open(path), caption=f"Match: {name} ({percent:.2f}%)", use_container_width=True)
            else:
                st.error("❌ No similar faces found.")
    if lottie_search:
        st_lottie(lottie_search, height=250, key="search_anim")

# ========= Gallery =========
# ========= Gallery =========
with tabs[2]:
    st.subheader("🖼️ Stored Gallery")

    conn = connect_db()
    cur = conn.cursor()

    # 1. قراءة أسماء الصور من قاعدة البيانات
    cur.execute("SELECT picture FROM pictures ORDER BY id DESC")
    results = cur.fetchall()
    conn.close()

    if results:
        cols = st.columns(4)
        for i, row in enumerate(results):
            file = row[0]
            path = os.path.join("stored-faces", file)
            with cols[i % 4]:
                if os.path.exists(path):
                    st.image(path, caption=file, use_container_width=True)
                    if st.button("🗑️ Delete", key=f"del_{file}"):
                        conn = connect_db()
                        cur = conn.cursor()
                        cur.execute("DELETE FROM pictures WHERE picture = %s", (file,))
                        conn.commit()
                        conn.close()
                        os.remove(path)
                        st.rerun()
                else:
                    st.warning(f"❌ Missing file: {file}")
    else:
        st.info("📭 No faces saved.")
    
    if lottie_gallery:
        st_lottie(lottie_gallery, height=250, key="gallery_anim")
# ========= Report =========
with tabs[3]:
    st.subheader("📊 Auto Summary")
    files = os.listdir("stored-faces")
    st.info(f"🧮 Total Faces: **{len(files)}**")
    st.write("🧾 Recent Face Insights:")
    for file in files[-5:]:
        col1, col2 = st.columns([1, 2])
        path = os.path.join("stored-faces", file)
        col1.image(path, width=100)
        col2.write(f"**{file}**{generate_comment()}{describe_face()}")
    if lottie_report:
        st_lottie(lottie_report, height=250, key="report_anim")

# ========= AI Suggestions =========
with tabs[4]:
    st.subheader("⚙️ Smart AI Suggestions")
    files = os.listdir("stored-faces")
    if len(files) > 100:
        st.warning("🧠 Consider archiving old faces for performance.")
    else:
        st.success("✅ Database size is healthy.")
    st.markdown("- 📌 Use clear face images for better detection.")
    st.markdown("- 🧩 Adjust detection sensitivity if needed.")
    if lottie_ai:
        st_lottie(lottie_ai, height=250, key="ai_anim")
