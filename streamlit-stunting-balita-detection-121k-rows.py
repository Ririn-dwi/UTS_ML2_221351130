import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

st.set_page_config(page_title="Deteksi Stunting Balita", layout="wide")

# Load Data
@st.cache_data
def load_data():
    df = pd.read_csv("data/stunting_data.csv")
    return df

df = load_data()

# Title
st.title("📊 Deteksi Stunting pada Balita")
st.markdown("Dataset berisi **121.000+ data** balita untuk mendeteksi apakah mengalami stunting atau tidak.")

# Data Overview
with st.expander("🔍 Lihat Info Dataset"):
    if st.checkbox("Tampilkan 5 Data Pertama"):
        st.dataframe(df.head())
    if st.checkbox("Tampilkan Info Umum"):
        st.write("Jumlah data:", df.shape)
        st.write("Tipe kolom:")
        st.write(df.dtypes)
    if st.checkbox("Cek Missing Values"):
        st.write(df.isnull().sum())

# EDA Section
st.subheader("📈 Exploratory Data Analysis (EDA)")
grafik = st.selectbox("Pilih Grafik", [
    "Distribusi Stunting", 
    "Usia vs Tinggi Badan", 
    "Jenis Kelamin vs Stunting", 
    "Boxplot Tinggi Badan", 
    "Korelasi Fitur"
])

if grafik == "Distribusi Stunting":
    fig, ax = plt.subplots()
    sns.countplot(x='stunting', data=df, ax=ax)
    st.pyplot(fig)

elif grafik == "Usia vs Tinggi Badan":
    fig, ax = plt.subplots()
    sns.scatterplot(x='usia_bulan', y='tinggi_badan', hue='stunting', data=df, ax=ax)
    st.pyplot(fig)

elif grafik == "Jenis Kelamin vs Stunting":
    fig, ax = plt.subplots()
    sns.countplot(x='jenis_kelamin', hue='stunting', data=df, ax=ax)
    st.pyplot(fig)

elif grafik == "Boxplot Tinggi Badan":
    fig, ax = plt.subplots()
    sns.boxplot(x='stunting', y='tinggi_badan', data=df, ax=ax)
    st.pyplot(fig)

elif grafik == "Korelasi Fitur":
    st.write("Matriks Korelasi:")
    fig, ax = plt.subplots()
    corr = df.select_dtypes(include='number').corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

# Modeling Section
st.subheader("⚙️ Modeling dan Prediksi")
if 'stunting' not in df.columns:
    st.error("Kolom 'stunting' tidak ditemukan.")
else:
    X = df.drop('stunting', axis=1)
    y = df['stunting']

    # Encode categorical columns
    X = pd.get_dummies(X)

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train Model
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    st.success(f"🎯 Akurasi Model: {acc*100:.2f}%")

    if st.checkbox("Tampilkan Evaluasi Model"):
        st.write("📉 Confusion Matrix:")
        st.write(confusion_matrix(y_test, y_pred))

        st.write("📝 Classification Report:")
        st.text(classification_report(y_test, y_pred))
