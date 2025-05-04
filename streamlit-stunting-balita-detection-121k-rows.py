import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

st.set_page_config(page_title="UTS ML2 - Deteksi Stunting", layout="wide")
st.title("📊 UTS ML2 - Deteksi Stunting pada Balita")

st.markdown("Aplikasi ini menggunakan model Machine Learning untuk mendeteksi potensi stunting berdasarkan data balita.")

# === Upload Dataset ===
uploaded_file = st.file_uploader("Unggah file dataset CSV", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("📄 Data Awal")
    st.write(df.head())

    # === EDA Sederhana ===
    st.subheader("📊 Exploratory Data Analysis (EDA)")
    st.write("Statistik Deskriptif")
    st.write(df.describe())

    st.write("Distribusi Label")
    if 'stunting' in df.columns:
        st.bar_chart(df['stunting'].value_counts())

    st.write("Heatmap Korelasi")
    fig, ax = plt.subplots()
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    # === Preprocessing dan Modeling ===
    st.subheader("🤖 Modeling: Random Forest Classifier")

    # Ganti nama kolom target sesuai dataset Anda
    target_col = "stunting"
    if target_col in df.columns:
        X = df.drop(columns=[target_col])
        y = df[target_col]

        # Membagi data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestClassifier()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=False)

        st.success(f"✅ Akurasi Model: {acc:.2f}")
        st.write("Confusion Matrix:")
        st.write(cm)
        st.text("Classification Report:")
        st.text(report)
    else:
        st.warning(f"Kolom target '{target_col}' tidak ditemukan dalam dataset.")

else:
    st.info("Silakan unggah file dataset CSV untuk mulai.")