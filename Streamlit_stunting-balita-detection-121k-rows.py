import streamlit as st
import pandas as pd

# Judul aplikasi
st.title("Deteksi Stunting Balita")

# Load dataset
df = pd.read_csv("data_balita.csv")

# Tampilkan data awal
with st.expander("Lihat Data Awal"):
    st.dataframe(df.head())

# Asumsikan kolom 'tinggi_badan' dan 'usia' tersedia
# Buat slider untuk memilih usia dan tinggi balita
st.header("Simulasi Deteksi Stunting Balita")
usia = st.slider("Masukkan Usia Balita (bulan)", 0, 60, 24)
tinggi = st.slider("Masukkan Tinggi Badan Balita (cm)", 40, 130, 80)

# Kriteria sederhana (misalnya WHO: anak usia 24 bulan stunting jika tinggi < 81 cm)
# Kamu bisa ganti dengan model ML jika sudah ada
def deteksi_stunting(usia, tinggi):
    if usia <= 24 and tinggi < 81:
        return "Stunting"
    elif usia > 24 and tinggi < 85:
        return "Stunting"
    else:
        return "Normal"

hasil = deteksi_stunting(usia, tinggi)

# Tombol deteksi
if st.button("Deteksi"):
    st.success(f"Hasil Deteksi: {hasil}")

# Statistik dataset
st.header("Statistik Dataset")
st.write(f"Jumlah Baris: {df.shape[0]}")
st.write(f"Jumlah Kolom: {df.shape[1]}")

# Distribusi jika kolom 'stunting' sudah tersedia
if 'stunting' in df.columns:
    st.subheader("Distribusi Data Stunting")
    st.write(df['stunting'].value_counts())