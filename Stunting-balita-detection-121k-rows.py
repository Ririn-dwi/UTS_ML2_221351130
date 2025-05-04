import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

st.title("📊 Stunting Toddler Detection App")
st.markdown("Prediksi status stunting pada balita berdasarkan fitur kesehatan dan demografi.")

# Load Data
@st.cache_data
def load_data():
    df = pd.read_csv("stunting.csv")
    return df

df = load_data()
st.subheader("👶 Data Sampel")
st.dataframe(df.head())

# -------------------- EDA --------------------
st.subheader("📈 Exploratory Data Analysis")
fig, ax = plt.subplots()
df['stunting_status'].value_counts().plot(kind='bar', ax=ax)
st.pyplot(fig)

fig2 = plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x='stunting_status', y='age_months')
st.pyplot(fig2)

col1, col2 = st.columns(2)
with col1:
    fig3 = plt.figure()
    sns.histplot(df['weight_kg'], kde=True)
    plt.title("Distribusi Berat Badan")
    st.pyplot(fig3)
with col2:
    fig4 = plt.figure()
    sns.histplot(df['height_cm'], kde=True)
    plt.title("Distribusi Tinggi Badan")
    st.pyplot(fig4)

# -------------------- Modeling --------------------
st.subheader("🤖 Model Pelatihan & Evaluasi")

if st.checkbox("Train Model"):
    st.write("🔁 Melatih model Random Forest...")

    X = df.drop('stunting_status', axis=1)
    y = df['stunting_status']

    # Encoding jika perlu
    X = pd.get_dummies(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    st.success(f"Akurasi Model: {acc:.2%}")

    cm = confusion_matrix(y_test, y_pred)
    st.text("Confusion Matrix:")
    st.dataframe(pd.DataFrame(cm, index=['Actual No', 'Actual Yes'], columns=['Pred No', 'Pred Yes']))

    st.text("Classification Report:")
    st.text(classification_report(y_test, y_pred))

    # Simpan model
    joblib.dump(model, "model.pkl")
    st.success("✅ Model disimpan ke model.pkl")

# -------------------- Prediksi --------------------
st.subheader("🔍 Prediksi Stunting Baru")
model = joblib.load("model.pkl")
with st.form("prediction_form"):
        age = st.number_input("Umur (bulan):", 0, 60, 12)
        weight = st.number_input("Berat (kg):", 0.0, 20.0, 8.0)
        height = st.number_input("Tinggi (cm):", 30.0, 120.0, 70.0)
        gender = st.selectbox("Jenis Kelamin", ['Laki-laki', 'Perempuan'])

        submitted = st.form_submit_button("Prediksi")
        if submitted:
            gender_num = 1 if gender == 'Laki-laki' else 0
            input_df = pd.DataFrame([[age, weight, height, gender_num]], columns=['age_months', 'weight_kg', 'height_cm', 'gender'])
            result = model.predict(input_df)[0]
            st.success(f"Hasil Prediksi: {'STUNTING' if result == 1 else 'Normal'}")