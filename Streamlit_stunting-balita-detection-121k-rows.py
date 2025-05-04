import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

st.set_page_config(page_title="Deteksi Stunting Balita", layout="wide")

@st.cache_data
def load_data():
    return pd.read_csv("data/data_balita.csv")

df = load_data()

st.title("📊 Deteksi Stunting pada Balita")
st.markdown("Dataset dengan **121K data** untuk mendeteksi apakah balita mengalami stunting.")

if st.checkbox("Tampilkan 5 Data Pertama"):
    st.dataframe(df.head())

if st.checkbox("Info Dataset"):
    st.write("Jumlah baris dan kolom:", df.shape)
    st.write("Tipe data:")
    st.write(df.dtypes)

if st.checkbox("Cek Missing Values"):
    st.write(df.isnull().sum())

st.subheader("📈 Visualisasi Data (EDA)")
grafik = st.selectbox("Pilih Grafik", ["Distribusi Stunting", "Usia vs Tinggi Badan", "Jenis Kelamin", "Tinggi Badan Boxplot", "Korelasi Fitur"])

if grafik == "Distribusi Stunting":
    sns.countplot(x='stunting', data=df)
    st.pyplot(plt.gcf())
    plt.clf()
elif grafik == "Usia vs Tinggi Badan":
    sns.scatterplot(x='usia_bulan', y='tinggi_badan', hue='stunting', data=df)
    st.pyplot(plt.gcf())
    plt.clf()
elif grafik == "Jenis Kelamin":
    sns.countplot(x='jenis_kelamin', hue='stunting', data=df)
    st.pyplot(plt.gcf())
    plt.clf()
elif grafik == "Tinggi Badan Boxplot":
    sns.boxplot(x='stunting', y='tinggi_badan', data=df)
    st.pyplot(plt.gcf())
    plt.clf()
elif grafik == "Korelasi Fitur":
    st.write("Korelasi antar fitur numerik:")
    corr = df.select_dtypes(include='number').corr()
    sns.heatmap(corr, annot=True, cmap='Blues')
    st.pyplot(plt.gcf())
    plt.clf()

st.subheader("⚙️ Modeling")
if 'stunting' not in df.columns:
    st.error("Kolom 'stunting' tidak ditemukan dalam dataset.")
else:
    X = df.drop('stunting', axis=1)
    y = df['stunting']
    X = pd.get_dummies(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    st.success(f"Akurasi Model: {acc:.2f}")
    if st.checkbox("Tampilkan Confusion Matrix dan Report"):
        cm = confusion_matrix(y_test, y_pred)
        st.write("Confusion Matrix:")
        st.write(cm)
        st.write("Classification Report:")
        st.text(classification_report(y_test, y_pred))
