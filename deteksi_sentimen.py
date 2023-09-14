import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

model_fraud = pickle.load(open('model_deteksi_komentar.sav','rb'))

tfidf = TfidfVectorizer

loaded_vec = TfidfVectorizer(decode_error="replace", vocabulary=set(pickle.load(open("new_selected_feature_tf-idf.sav", "rb"))))

def main():
    st.title("Deteksi Sentimen")
    message = st.text_area("Masukan")

    if st.button("Deteksi Sentimen Masukan"):
        predict_fraud = model_fraud.predict(loaded_vec.fit_transform([message]))
    
        if predict_fraud == 0:
            fraud_detection = 'Masukan kamu terindikasi hatespeech'
        elif predict_fraud == 1:
            fraud_detection = 'Terima kasih masukannya, sangat membangun'
        else:
            fraud_detection = 'inputanmu salah'
        st.success(fraud_detection)

if __name__ == "__main__":
    main()
