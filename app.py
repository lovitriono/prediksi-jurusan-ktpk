import streamlit as st
import numpy as np
import joblib

# =========================
# LOAD MODEL & ENCODER
# =========================
model = joblib.load("model_rf.pkl")
scaler = joblib.load("scaler.pkl")
label_encoders = joblib.load("label_encoders.pkl")

st.set_page_config(page_title="Prediksi Jurusan SMA", page_icon="🎓", layout="centered")

# =========================
# TITLE
# =========================
st.title("🎓 Aplikasi Prediksi Jurusan Siswa")
st.write("Masukkan data siswa untuk mendapatkan rekomendasi jurusan berdasarkan model Regresi Linier.")
st.write("Lovi Triono - SMKN 1 Kota Bengkulu Peserta Mikrokredential AI dan Data Science KPTK")

# =========================
# INPUT FORM
# =========================
st.subheader("📌 Input Data Siswa")

col1, col2 = st.columns(2)

with col1:
    nilai_matematika = st.number_input("Nilai Matematika", 0, 100)
    nilai_ipa = st.number_input("Nilai IPA", 0, 100)
    minat_ipa = st.selectbox("Minat IPA", label_encoders["minat_ipa"].classes_)

with col2:
    nilai_bahasa_inggris = st.number_input("Nilai Bahasa Inggris", 0, 100)
    nilai_ips = st.number_input("Nilai IPS", 0, 100)
    minat_ips = st.selectbox("Minat IPS", label_encoders["minat_ips"].classes_)

minat_bahasa = st.selectbox("Minat Bahasa", label_encoders["minat_bahasa"].classes_)
ekonomi_keluarga = st.selectbox("Ekonomi Keluarga", label_encoders["ekonomi_keluarga"].classes_)
tipe_sekolah = st.selectbox("Tipe Sekolah", label_encoders["tipe_sekolah"].classes_)

# =========================
# ENCODING INPUT
# =========================
def encode(col, val):
    return label_encoders[col].transform([val])[0]

encoded_data = [
    nilai_matematika,
    nilai_bahasa_inggris,
    nilai_ipa,
    nilai_ips,
    encode("minat_ipa", minat_ipa),
    encode("minat_ips", minat_ips),
    encode("minat_bahasa", minat_bahasa),
    encode("ekonomi_keluarga", ekonomi_keluarga),
    encode("tipe_sekolah", tipe_sekolah)
]

# =========================
# PREDIKSI
# =========================
if st.button("🔍 Prediksi Jurusan"):
    X = np.array([encoded_data])
    X_scaled = scaler.transform(X)

    pred = model.predict(X_scaled)[0]
    jurusan = label_encoders["jurusan_output"].inverse_transform([pred])[0]

    st.success(f"🎉 Jurusan yang direkomendasikan: **{jurusan}**")

    st.info("Prediksi didasarkan pada nilai akademik, minat siswa, ekonomi keluarga, dan tipe sekolah.")

