import streamlit as st
import pandas as pd

st.set_page_config(page_title="Deteksi Stunting - RIRIN DWI NURHASANAH")

st.title("Aplikasi Deteksi Stunting Balita")
st.subheader("Oleh: RIRIN DWI NURHASANAH")

# Load dataset
df = pd.read_csv("data_balita.csv")

# Tambahkan kolom 'stunting' jika belum ada
if 'tinggi_badan' in df.columns and 'usia' in df.columns:
    df['stunting'] = df.apply(lambda row: 1 if (row['usia'] <= 24 and row['tinggi_badan'] < 81) or 
                                           (row['usia'] > 24 and row['tinggi_badan'] < 85) else 0, axis=1)

# Tampilkan data
st.subheader("Contoh Data")
st.dataframe(df.head())

# Distribusi target
if 'stunting' in df.columns:
    st.subheader("Distribusi Kategori Stunting")
    st.bar_chart(df['stunting'].value_counts())

# Input user
st.header("Simulasi Deteksi")
usia = st.slider("Usia Balita (bulan)", 0, 60, 24)
tinggi = st.slider("Tinggi Badan (cm)", 40, 130, 80)

def prediksi(usia, tinggi):
    if usia <= 24 and tinggi < 81:
        return "Stunting"
    elif usia > 24 and tinggi < 85:
        return "Stunting"
    else:
        return "Normal"

if st.button("Deteksi"):
    hasil = prediksi(usia, tinggi)
    st.success(f"Hasil Deteksi: {hasil}")
