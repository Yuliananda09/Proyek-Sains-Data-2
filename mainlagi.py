import pickle
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

st.markdown(
    "<h1 style='text-align: center;'>Klasifikasi Tumor Otak Menggunakan Model Naive Bayes</h1>", unsafe_allow_html=True
)
st.markdown(
    "<h4 style='text-align: center;'>Yulia Nanda Ivne Fara | 210411100158 | PSD - B</h4>", unsafe_allow_html=True
)

# load dataset
dataset = pd.read_csv('dataset_baru_nb.csv')

# split dataset menjadi data training dan data testing 
fitur = dataset.drop(columns=['Grade'], axis =1)
target = dataset['Grade']
fitur_train, fitur_test, target_train, target_test = train_test_split(fitur, target, test_size = 0.2, random_state=42)

# normalisasi dataset 
# memanggil kembali model normalisasi zscore dari file pickle
with open('zcorescaler_baru.pkl', 'rb') as file_normalisasi:
    zscore = pickle.load(file_normalisasi)

zscoretraining = zscore.transform(fitur_train)
zscoretesting = zscore.transform(fitur_test)

# implementasi data pda model
with open('best_nb_model.pkl', 'rb') as file_model:
    model_nb = pickle.load(file_model)

model_nb.fit(zscoretraining, target_train)
prediksi_target = model_nb.predict(zscoretesting)

# Input status mutasi IDH1
IDH1 = int(st.radio(
    "Apakah Anda telah menjalani pengujian untuk mutasi IDH1? (Pengujian mutasi IDH1 dapat membantu mengidentifikasi adanya perubahan genetik yang berperan dalam glioma. Ini penting untuk merencanakan pengobatan yang sesuai.)", ["0", "1"]
))

# Input status mutasi PTEN
Age_at_diagnosis = st.number_input("Input umur anda")

# Input status mutasi CIC
CIC = int(st.radio(
    "Apakah Anda memiliki mutasi CIC? (Mutasi CIC pada glioma dapat memengaruhi regulasi ekspresi gen dan berkontribusi pada perkembangan kanker. Ini dapat berpengaruh pada karakteristik dan agresivitas tumor.)", ["0", "1"]
))

# Input status mutasi CIC
NOTCH1 = int(st.radio(
    "Apakah Anda memiliki mutasi NOTCH1? (Mutasi NOTCH1 pada glioma dapat memengaruhi regulasi ekspresi gen dan berkontribusi pada perkembangan kanker. Ini dapat berpengaruh pada karakteristik dan agresivitas tumor.)", ["0", "1"]
))

if st.button('Cek Status'):
    if IDH1 is not None and Age_at_diagnosis is not 0 and CIC is not None and NOTCH1 is not None:
        # Format input sesuai dengan model dan fitur-fitur yang digunakan
        prediksi = model_nb.predict([[IDH1, Age_at_diagnosis, CIC, NOTCH1]])
        
        if prediksi == 0.0:
            st.success("Kondisi stabil (tanpa kekambuhan)!")
        elif prediksi == 1.0:
            st.warning("Kekambuhan atau hasil yang kurang menguntungkan (termasuk kematian)")
    else:
        st.text('Data tidak boleh kosong. Harap isi semua kolom.')
