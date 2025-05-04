import streamlit as st
import pandas as pd

st.title("Stunting Toddler Detection")

# Load dataset dari file lokal yang benar
df = pd.read_csv("data_balita.csv")

# Tampilkan jumlah data
st.subheader("Jumlah Data")
st.write(f"Total baris (rows): {df.shape[0]}")
st.write(f"Total kolom (columns): {df.shape[1]}")

# Tampilkan sebagian data
st.subheader("Contoh Data")
st.dataframe(df.head())

# Opsional: tampilkan distribusi target
if 'stunting' in df.columns:
    st.subheader("Distribusi Target (Stunting)")
    st.write(df['stunting'].value_counts())
