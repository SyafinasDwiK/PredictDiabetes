import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
from sklearn.preprocessing import MinMaxScaler


st.set_page_config(page_title="Diabetes Prediction", layout="wide")
# Create menu
selected = option_menu(
    menu_title=None,
    options=["Home", "Data Visualisation", "Prediction", "Training Result", "Diabetes First Treatment"],
    icons=["house", "book", "calculator", "sliders", "plus-circle"],
    menu_icon="cast",
    default_index=0,
    orientation="horizontal",
)

row0_spacer1, row0_1, row0_spacer2 = st.columns((0.1, 3.2, 0.1))
row1_spacer1, row1_1, row1_spacer2, row1_2 = st.columns((0.1, 1.5, 0.1, 1.5))
row0_spacer3, row3_0, row0_spacer4 = st.columns((0.1, 3.2, 0.1))

# Load dataset
df = pd.read_csv('Data/diabetes_prediction_dataset_train_test.csv')
# Kelompok usia
age_grup = []
for i in df['age']:
    if i >= 17 and i <= 25:
        age_grup.append('Remaja Akhir')
    elif i >= 26 and i <= 35:
        age_grup.append('Dewasa Awal')
    elif i >= 36 and i <= 45:
        age_grup.append('Dewasa Akhir')
    elif i >= 46 and i <= 55:
        age_grup.append('Lansia Awal')
    elif i >= 56 and i <= 65:
        age_grup.append('Lansia Akhir')
    else:
        age_grup.append('Manula')
df['AgeGrup'] = age_grup
# Kelompok BMI
BMI_grup = []
for i in df['bmi']:
    if i >= 0 and i <= 18.5:
        BMI_grup.append('Kurus')
    elif i >= 18.6 and i <= 22.9:
        BMI_grup.append('Normal')
    elif i >= 23 and i <= 24.9:
        BMI_grup.append('Gemuk')
    elif i >= 25 and i <= 29.9:
        BMI_grup.append('Obesitas')
    else:
        BMI_grup.append('Obesitas II')            
df['BMIGrup'] = BMI_grup

# Daftar model yang tersedia
available_models = {
    'Logistic Regression': 'new_model/Logistic_Regression.pkl',
    'K-NN': 'new_model/K-NN.pkl',
    'Decision Tree': 'new_model/Decision_Tree.pkl',
    'Random Forest': 'new_model/Random_Forest.pkl',
    'SVM': 'new_model/SVM.pkl',
    'Gradient Boosting': 'new_model/Gradient_Boosting.pkl',
    'XGBoost': 'new_model/XGBoost.pkl',
    'LightGBM': 'new_model/LightGBM.pkl',
    'Naive Bayes': 'new_model/Naive_Bayes.pkl'
}
# Function to preprocess and predict
def preprocess_and_predict(data, scaler):
    # Convert to DataFrame to match feature names
    columns = ['age', 'hypertension', 'heart_disease', 'bmi', 'HbA1c_level', 'blood_glucose_level', 
               'gender_Female', 'gender_Male', 'gender_Other', 
               'smoking_history_current', 'smoking_history_ever', 'smoking_history_former', 'smoking_history_never','smoking_history_notcurrent']
    
    data_df = pd.DataFrame(data, columns=columns)

    # Scale continuous columns
    continuous_columns = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']
    data_df[continuous_columns] = scaler.transform(data_df[continuous_columns])
    
    # Predict using the loaded model
    prediction = loaded_model.predict(data_df)
    return prediction

# Handle selected option
if selected == "Home":
    row0_1.title("Diabetes Prediction Web ヾ(≧▽≦*)o")
    with row0_1:
        st.markdown(
            "Website ini berguna untuk memprediksi kemungkinan seseorang menderita diabetes berdasarkan beberapa fitur yang dimasukkan. Website ini menggunakan dataset diabetes dari Kaggle untuk melakukan prediksi. Dengan memasukkan fitur yang relevan, seperti kadar gula darah, tekanan darah, dan sebagainya, website ini dapat memberikan prediksi yang cukup akurat mengenai kemungkinan seseorang menderita diabetes. Website ini sangat bermanfaat bagi orang-orang yang ingin mengetahui apakah mereka berisiko terkena diabetes atau tidak, sehingga dapat memperbaiki pola makan dan gaya hidup mereka untuk mencegah terjadinya penyakit diabetes."
        )
        st.markdown('Dataset : https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset')
        st.write(df.head())
        st.markdown('Atribut Dataset :')
        st.markdown("1. Gender: Merupakan jenis kelamin pasien ")
        st.markdown("2. Age: Merupakan usia pasien")
        st.markdown("3. Hypertension: Merupakan tekanan darah pasien")
        st.markdown("4. Heart Disease: Merupakan riwayat penyakit jantung pasien")
        st.markdown("5. Smoking History: Merupakan riwayat merokok pasien")
        st.markdown("6. BMI: Merupakan Body Mass Index pasien")
        st.markdown("7. Hemoglobin A1c: Merupakan persentase hemoglobin dalam darah yang terikat dengan glukosa pasien")
        st.markdown("8. Blood Glucose Level: Merupakan kadar gula darah pasien")
        st.markdown("9. Diabetes: Merupakan hasil diagnosis pasien, 0 berarti tidak terkena diabetes, 1 berarti terkena diabetes")

elif selected == "Data Visualisation":
    # Data Visualisasi dengan plotly
    with row1_1:
        st.subheader('Pilih fitur yang ingin ditampilkan histogramnya')
        fitur = st.selectbox('Fitur', ('gender', 'age', 'hypertension', 'heart_disease', 'smoking_history', 'bmi', 'HbA1c_level', 'blood_glucose_level','diabetes'))
        fig = px.histogram(df, x=fitur, color='diabetes', marginal='box', hover_data=df.columns, color_discrete_sequence=["#6A8EEB", "#FDBFD5", "#AD81B8"])
        st.plotly_chart(fig)
        fig = px.histogram(df, x='diabetes', color='gender', barmode='group', hover_data=df.columns, color_discrete_sequence=["#6A8EEB", "#FDBFD5", "#AD81B8"])
        fig.update_layout(title='Jumlah pasien per kelompok jenis kelamin', xaxis_title='diabetes', yaxis_title='Jumlah', font=dict(size=15))
        st.plotly_chart(fig)
        st.markdown(
            'Data menunjukkan bahwa diabetes paling banyak diderita oleh pasien wanita, dengan jumlah pasien wanita mencapai 4228 dan pasien pria sebanyak 3836. Perlu dicatat bahwa risiko terkena diabetes tidak selalu berkorelasi dengan jenis kelamin, sehingga pemeriksaan rutin dan penerapan gaya hidup sehat tetap dianggap penting.'
        )
        fig = px.histogram(df, x='diabetes', color='AgeGrup', barmode='group', hover_data=df.columns, color_discrete_sequence=["#AD81B8","#6A8EEB","#98b4fe","#FDECF0","#FDBFD5","#E53E6C"])
        fig.update_layout(title='Jumlah pasien per kelompok usia', xaxis_title='diabetes', yaxis_title='Jumlah', font=dict(size=15))
        st.plotly_chart(fig)
        st.markdown(
            "Dari data yang dianalisis, terlihat bahwa kelompok usia lebih dari 65 tahun atau manula memiliki jumlah pasien diabetes terbanyak, mencapai 3447 orang. Sementara itu, kelompok remaja akhir yaitu 17-25 tahun memiliki jumlah pasien diabetes yang paling sedikit, mencapai 94 orang. Hal ini menunjukkan bahwa usia masih menjadi faktor penting dalam kejadian diabetes pada pasien."
        )
    with row1_2:
        st.subheader('Pilih fitur yang ingin ditampilkan scatter plotnya')
        fitur1 = st.selectbox('Fitur 1', ('gender', 'age', 'hypertension', 'heart_disease', 'smoking_history', 'bmi', 'HbA1c_level', 'blood_glucose_level','diabetes'))
        fitur2 = st.selectbox('Fitur 2', ('gender', 'age', 'hypertension', 'heart_disease', 'smoking_history', 'bmi', 'HbA1c_level', 'blood_glucose_level','diabetes'))
        fig = px.scatter(df, x=fitur1, y=fitur2, color='diabetes', hover_data=df.columns)
        st.plotly_chart(fig)
        fig = px.histogram(df, x='diabetes', color='diabetes', hover_data=df.columns, color_discrete_sequence=["#AD81B8","#6A8EEB","#98b4fe","#FDECF0","#FDBFD5","#E53E6C"])
        fig.update_layout(title='Jumlah Pasien Diabetes', xaxis_title='diabetes', yaxis_title='Jumlah', font=dict(size=15))
        st.plotly_chart(fig)
        st.markdown(
            'Mari kita lihat grafik di atas. Dari grafik tersebut, terlihat bahwa mayoritas orang yang dianalisis dalam studi ini tidak terkena diabetes. Namun, terdapat sejumlah kecil orang yang terdiagnosis dengan diabetes, yaitu hanya 8064 orang dari keseluruhan. Ini menunjukkan bahwa diabetes mungkin masih merupakan masalah kesehatan yang signifikan, namun masih mempengaruhi sebagian kecil populasi.'
        )
        fig = px.histogram(df, x='diabetes', color='BMIGrup', barmode='group', hover_data=df.columns, color_discrete_sequence=["#AD81B8","#6A8EEB","#98b4fe","#FDECF0","#FDBFD5","#E53E6C"])
        fig.update_layout(title='Jumlah pasien per kelompok BMI', xaxis_title='diabetes', yaxis_title='Jumlah', font=dict(size=15))
        st.plotly_chart(fig)
        st.markdown(
            'Dari hasil analisis data, terlihat bahwa kondisi BMI memiliki korelasi yang kuat dengan kemungkinan seseorang terkena diabetes. Kelompok BMI yang paling berisiko adalah yang memiliki BMI lebih dari 30 (Obesitas II), dengan jumlah 4109 orang yang terkena diabetes. Sementara itu, kelompok BMI 25-29,9 (Obesitas) juga memiliki risiko yang cukup tinggi dengan jumlah 3124 orang yang terkena diabetes. Grafik ini menunjukkan betapa pentingnya menjaga kesehatan dan berat badan yang sehat untuk mencegah risiko terkena diabetes.'
        )
elif selected == "Prediction":
    with row0_1:
        st.subheader('Masukkan Data ༼ つ ◕_◕ ༽つ')

        selected_model_name = st.selectbox("Pilih Model",list(available_models.keys()))
        model_path = available_models[selected_model_name]
        with open(model_path, 'rb') as file:
            loaded_model = joblib.load(file)

        st.success(f"Model '{selected_model_name}' berhasil dimuat!")

    with row1_1:
        gender = st.selectbox("Jenis Kelamin", [" ","Pria", "Wanita"])
        gender_onehot = {
            "Pria": [0, 1, 0],     # Female=0, Male=1, Other=0
            "Wanita": [1, 0, 0],   # Female=1, Male=0, Other=0
            " ": [0, 0, 1]   # Female=0, Male=0, Other=1
        }[gender]
        
        age = st.number_input("Usia", min_value=0, max_value=120, value=30)
        hypertension = st.selectbox("Riwayat Hipertensi", ["Tidak", "Ya"])
        hypertension_encoded = 1 if hypertension == "Ya" else 0

        heart_disease = st.selectbox("Riwayat Penyakit Jantung", ["Tidak", "Ya"])
        heart_disease_encoded = 1 if heart_disease == "Ya" else 0


    with row1_2:
        
        smoking_history = st.selectbox("Riwayat Merokok", ["Saat Ini", "Pernah", "Mantan Perokok", "Tidak Pernah", "Tidak Saat Ini"])
        smoking_onehot = {
            "Saat Ini": [1, 0, 0, 0, 0],
            "Pernah": [0, 1, 0, 0, 0],
            "Mantan Perokok": [0, 0, 1, 0, 0],
            "Tidak Pernah": [0, 0, 0, 1, 0],
            "Tidak Saat Ini": [0, 0, 0, 0, 1]

        }[smoking_history]
        bmi = st.number_input("BMI (Body Mass Index)", min_value=0.0, max_value=70.0, value=25.0, step=0.1)
        hba1c_level = st.number_input("HbA1c Level (%)", min_value=0.0, max_value=20.0, value=5.5, step=0.1)
        blood_glucose_level = st.number_input("Blood Glucose Level (mg/dL)", min_value=0, max_value=300, value=100)

    with row3_0:
        button = st.button('Prediksi')
        if button:
           # Membuat array data user
            data = [[
                        age, hypertension_encoded, heart_disease_encoded, bmi, hba1c_level, blood_glucose_level,
                        *gender_onehot, *smoking_onehot
                    ]]

            # Load scaler for preprocessing
            scaler = joblib.load('new_model/scaler.pkl')
            
            # Predict
            pred = preprocess_and_predict(data, scaler)

            if pred[0] == 1:
                st.error("**Hasil Prediksi:** Anda berpotensi memiliki diabetes. Segera konsultasikan dengan dokter.")
            
            else:
                st.success("**Hasil Prediksi:** Anda tidak memiliki indikasi diabetes. Tetap jaga kesehatan Anda!")
                st.balloons()
                
elif selected == "Training Result":
    with row0_1:
        st.title("Training Metrics (っ °Д °;)っ")
    
    with row1_1:
        # Menampilkan hasil  klasifikasi LightGBM
        st.subheader("Akurasi Klasifikasi LightGBM")
        st.image("img/classification_lgbm.jpeg", caption="Nilai Akurasi Klasifikasi LightGBM")
        st.markdown("""
            Gambar ini menampilkan classification report yang berisi evaluasi performa model klasifikasi berdasarkan metrik precision, recall, f1-score, dan support untuk setiap kelas.

            Kelas 0 (negatif):

            Precision: 0.97 menunjukkan bahwa 97% prediksi kelas 0 benar.
            Recall: 1.00 menunjukkan bahwa semua sampel kelas 0 berhasil diidentifikasi oleh model.
            F1-score: 0.98 adalah kombinasi harmonis antara precision dan recall yang menunjukkan performa sangat baik pada kelas ini.
            Support: 26,062 sampel termasuk dalam kelas 0.
            Kelas 1 (positif):

            Precision: 0.97 menunjukkan bahwa 97% prediksi kelas 1 benar.
            Recall: 0.68 menunjukkan bahwa hanya 68% sampel kelas 1 yang berhasil diidentifikasi dengan benar oleh model.
            F1-score: 0.80 sebagai keseimbangan antara precision dan recall, mencerminkan performa yang baik tetapi masih bisa ditingkatkan untuk kelas ini.
            Support: 2,438 sampel termasuk dalam kelas 1.
            Akurasi Keseluruhan:
            Akurasi model sebesar 97%, menunjukkan bahwa model mampu memprediksi dengan benar 97% dari total 28,500 sampel.

            Rata-Rata (Macro dan Weighted Avg):

            Macro Avg: Precision 0.97, Recall 0.84, dan F1-score 0.89 menunjukkan rata-rata sederhana antar kelas.
            Weighted Avg: Precision, Recall, dan F1-score masing-masing 0.97, memberikan bobot lebih besar untuk kelas dengan jumlah sampel lebih banyak (kelas 0).
            Secara keseluruhan, model memiliki performa sangat baik untuk kelas mayoritas (kelas 0) tetapi perlu ditingkatkan untuk kelas minoritas (kelas 1), terutama pada metrik recall yang lebih rendah. Hal ini mengindikasikan bahwa model mungkin kurang optimal dalam mendeteksi sampel positif.
        """)  
        
        # Menampilkan confusion matrix
        st.subheader("Confusion Matrix")
        st.image("img/cm_test.png", caption="Confusion Matrix")
        st.markdown("""

            Confusion Matrix menunjukkan jumlah prediksi yang benar dan salah, memberikan informasi visual tentang kesalahan klasifikasi model.
            Berdasarkan confusion matrix:

            True Positive (TP): 26018 (prediksi positif yang benar).

            False Positive (FP): 44 (prediksi positif yang salah).

            True Negative (TN): 780 (prediksi negatif yang benar).

            False Negative (FN): 1658 (prediksi negatif yang salah).

            Rumus untuk menghitung presisi dan recall:

            Presisi = TP / (TP + FP) = 26018 / (26018 + 44) ≈ 0.974 (97.4%).

            Recall = TP / (TP + FN) = 26018 / (26018 + 1658) ≈ 0.680 (68%).
            """)  

        
    with row1_2:
        # Menampilkan grafik training dan testing
        st.subheader("Grafik ROC")
        st.image("img/model_train.png", caption="Grafik Receiver Operating Characteristic (ROC)")  
        st.markdown("""
                    
            Grafik ini menunjukkan Receiver Operating Characteristic (ROC) curve untuk model LightGBM. ROC curve adalah representasi visual yang menggambarkan performa model klasifikasi pada berbagai nilai threshold. Sumbu x menunjukkan False Positive Rate (FPR), sedangkan sumbu y menunjukkan True Positive Rate (TPR) atau sensitivitas.

            Kurva ROC berwarna oranye terlihat mendekati sudut kiri atas grafik, yang menandakan performa model yang sangat baik. Area di bawah kurva (AUC - Area Under the Curve) sebesar 0.98 menunjukkan bahwa model memiliki kemampuan diskriminasi yang sangat tinggi dalam membedakan kelas positif dan negatif. Semakin mendekati nilai AUC ke 1, semakin baik performa model. Dalam hal ini, nilai AUC sebesar 0.98 menunjukkan bahwa model LightGBM mampu memprediksi dengan akurasi yang hampir sempurna.

            Selain itu, garis diagonal biru pada grafik menunjukkan baseline random guessing (prediksi acak) dengan AUC = 0.5. Performa model yang jauh di atas garis ini menegaskan bahwa model memiliki kemampuan prediksi yang jauh lebih baik dibandingkan dengan tebakan acak.


        """)
elif selected == "Diabetes First Treatment":
    with row0_1:
        st.title("Treatment -(￣▽￣)~*")

    with row1_1:
        # Pendahuluan
        st.markdown("""
        Diabetes merupakan kondisi kesehatan serius yang membutuhkan perhatian dan penanganan sejak dini. 
        Saat ini, diabetes tipe 1 belum dapat dicegah, namun berbagai langkah efektif tersedia untuk mencegah diabetes tipe 2, 
        serta mengurangi risiko komplikasi dan kematian dini yang dapat terjadi pada semua jenis diabetes.
        """)

        # Pencegahan dan Gaya Hidup Sehat
        st.subheader("Pencegahan dan Gaya Hidup Sehat")
        st.markdown("""
        Pola hidup sehat menjadi kunci utama dalam mengelola diabetes dan menjaga kualitas hidup, baik untuk individu yang telah terdiagnosis maupun yang berisiko. Beberapa langkah penting meliputi:
        - **Olahraga teratur:** Aktivitas fisik membantu mengontrol kadar gula darah dan meningkatkan kebugaran.
        - **Pola makan sehat:** Konsumsi makanan bergizi seimbang dengan mengurangi asupan gula dan lemak jenuh.
        - **Hindari merokok:** Kebiasaan ini dapat meningkatkan risiko komplikasi diabetes, terutama penyakit jantung dan pembuluh darah.
        - **Pengendalian tekanan darah dan kadar lipid:** Mengurangi risiko komplikasi kardiovaskular.
        """)

    with row1_2:
        # Pentingnya Diagnosis Dini
        st.subheader("Pentingnya Diagnosis Dini")
        st.markdown("""
        Diagnosis dini sangat penting untuk memulai pengelolaan diabetes yang tepat. Semakin lama seseorang hidup dengan diabetes yang tidak terdiagnosis dan tidak diobati, semakin buruk dampaknya terhadap kesehatan. Oleh karena itu, akses mudah ke alat diagnostik dasar seperti tes gula darah perlu tersedia di layanan kesehatan primer. Setelah diagnosis, pasien membutuhkan penilaian berkala oleh spesialis untuk mendeteksi komplikasi.
        """)

        # Intervensi untuk Hasil yang Lebih Baik
        st.subheader("Intervensi untuk Hasil yang Lebih Baik")
        st.markdown("""
        Serangkaian langkah hemat biaya dapat membantu meningkatkan kualitas hidup penderita diabetes, termasuk:
        - **Pengendalian gula darah:** Mengombinasikan pola makan, aktivitas fisik, dan bila perlu, obat-obatan.
        - **Pengelolaan tekanan darah dan lipid:** Mengurangi risiko komplikasi kardiovaskular dan masalah lainnya.
        - **Skrining rutin:** Pemeriksaan mata, ginjal, dan kaki secara berkala untuk mendeteksi dan mengobati kerusakan sejak dini.
        """)
    
    with row3_0:
        # Penutup
        st.markdown("""
        Dengan pendekatan yang terintegrasi dan dukungan dari berbagai pihak, penanganan diabetes dapat dilakukan secara efektif untuk mendukung kualitas hidup yang lebih baik bagi pasien.
        """)

