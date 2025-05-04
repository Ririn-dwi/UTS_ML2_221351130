# streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# Konfigurasi halaman
st.set_page_config(page_title="Deteksi Stunting Balita", layout="wide")
st.title("📊 UTS ML2 - Deteksi Stunting pada Balita")
st.markdown("Aplikasi ini menggunakan Machine Learning untuk mendeteksi potensi stunting pada balita berdasarkan dataset CSV yang diunggah.")

# Upload file CSV
uploaded_file = st.file_uploader("📁 Unggah file dataset (.csv)", type="csv")

if uploaded_file is not None:
    # Load data
    df = pd.read_csv(uploaded_file)

    # Cek apakah kolom target ada
    target_col = "stunting"
    if target_col not in df.columns:
        st.error(f"Kolom target '{target_col}' tidak ditemukan di dataset. Mohon pastikan kolom target bernama 'stunting'.")
        st.stop()

    st.subheader("📄 Data Awal")
    st.write(df.head())

    # Exploratory Data Analysis (EDA)
    st.subheader("📊 Exploratory Data Analysis (EDA)")

    st.markdown("**1. Statistik Deskriptif**")
    st.write(df.describe())

    st.markdown("**2. Distribusi Target (Stunting)**")
    st.bar_chart(df[target_col].value_counts())

    st.markdown("**3. Korelasi Fitur (Heatmap)**")
    fig1, ax1 = plt.subplots()
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax1)
    st.pyplot(fig1)

    st.markdown("**4. Histogram Setiap Fitur**")
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    df.drop(columns=[target_col]).hist(ax=ax2, figsize=(12, 6), bins=20)
    st.pyplot(fig2)

    st.markdown("**5. Boxplot Fitur terhadap Target**")
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    for col in numeric_cols:
        if col != target_col:
            sns.boxplot(x=target_col, y=col, data=df, ax=ax3)
            st.pyplot(fig3)
            break  # Tampilkan 1 contoh saja untuk efisiensi

    # Preprocessing
    st.subheader("⚙️ Preprocessing & Modeling")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Handle kategori non-numerik jika ada
    X = pd.get_dummies(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Modeling
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Evaluation
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=False)

    st.subheader("✅ Evaluasi Model")
    st.write(f"**Akurasi:** {acc:.2%}")

    st.markdown("**Confusion Matrix**")
    fig_cm, ax_cm = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", ax=ax_cm)
    st.pyplot(fig_cm)

    st.markdown("**Classification Report**")
    st.text(report)

else:
    st.info("Silakan unggah file CSV untuk memulai.")
