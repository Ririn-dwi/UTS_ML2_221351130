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
st.markdown("Aplikasi ini menggunakan Machine Learning untuk mendeteksi potensi stunting pada balita.")

# Upload file CSV
uploaded_file = st.file_uploader("📁 Unggah file dataset (.csv)", type="csv")

if uploaded_file is not None:
    # Load data
    df = pd.read_csv(uploaded_file)

    # Kolom target
    target_col = "stunting"
    if target_col not in df.columns:
        st.error(f"❌ Kolom target '{target_col}' tidak ditemukan di dataset.")
        st.stop()

    # Tampilkan data awal
    st.subheader("📄 Data Awal")
    st.write(df.head())

    # Exploratory Data Analysis
    st.subheader("📊 Exploratory Data Analysis (EDA)")

    st.markdown("**1. Statistik Deskriptif**")
    st.write(df.describe())

    st.markdown("**2. Distribusi Label 'Stunting'**")
    st.bar_chart(df[target_col].value_counts())

    st.markdown("**3. Heatmap Korelasi**")
    fig1, ax1 = plt.subplots()
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", ax=ax1)
    st.pyplot(fig1)

    st.markdown("**4. Histogram Fitur Numerik**")
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    if len(num_cols) > 1:
        fig2, axs2 = plt.subplots(len(num_cols)-1, 1, figsize=(10, 5*(len(num_cols)-1)))
        for i, col in enumerate([c for c in num_cols if c != target_col]):
            sns.histplot(df[col], bins=20, ax=axs2[i], kde=True)
            axs2[i].set_title(f"Distribusi: {col}")
        st.pyplot(fig2)

    st.markdown("**5. Boxplot Contoh Fitur vs Target**")
    example_col = [col for col in num_cols if col != target_col][0]
    fig3, ax3 = plt.subplots()
    sns.boxplot(x=target_col, y=example_col, data=df, ax=ax3)
    st.pyplot(fig3)

    # Preprocessing dan Modeling
    st.subheader("🤖 Modeling - Random Forest Classifier")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Encode kolom kategorikal jika ada
    X = pd.get_dummies(X)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Evaluasi
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    st.success(f"✅ Akurasi Model: {acc:.2%}")

    st.markdown("**Confusion Matrix**")
    fig_cm, ax_cm = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
    st.pyplot(fig_cm)

    st.markdown("**Classification Report**")
    st.text(report)

else:
    st.info("Silakan unggah file CSV terlebih dahulu untuk mulai.")
