# PredictDiabetes
Memprediksi kemungkinan seseorang menderita diabetes berdasarkan beberapa fitur yang relevan, seperti kadar gula darah, tekanan darah, dan sebagainya, website ini dapat memberikan prediksi yang cukup akurat mengenai kemungkinan seseorang menderita diabetes.

# 🚀 Machine Learning Model Training & Deployment

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red.svg)
![Status](https://img.shields.io/badge/Status-Development-green)

Proyek ini berisi implementasi proses **training model Machine Learning** serta **deployment sederhana menggunakan Streamlit** untuk menampilkan hasil prediksi secara interaktif.

---

# 📂 Project Structure
├── TrainingLaporan2FixIni.ipynb # Notebook training model

├── new_model/ # Folder model hasil training (.pkl)

├── requirements.txt # Dependency project

└── web.py # Aplikasi Streamlit


---

# 🧠 1. Training Model

File: `TrainingLaporan2FixIni.ipynb`

Notebook ini digunakan untuk melakukan proses:

- 📥 Load dataset  
- 🧹 Data preprocessing  
- 🤖 Training model Machine Learning  
- 📊 Evaluasi performa model  
- 💾 Menyimpan model ke file `.pkl`  

📌 **Catatan:**  
Model yang sudah dilatih akan disimpan dan digunakan oleh aplikasi web tanpa perlu training ulang.

---

# 📦 2. Model Storage

Folder: `new_model/`

Berisi file model hasil training:
Decision_Tree, XGBoost, SVM, Random_Forest, Naive_Bayes, Logistic_Regression, LightGBM, K-NN, Gradient_Boosting


✔ Setiap file `.pkl` merupakan model siap pakai untuk prediksi.

---

# 📋 3. Requirements

File: `requirements.txt`

Digunakan untuk menginstall semua library yang dibutuhkan.

### Install dependencies:
```bash
pip install -r requirements.txt
```
---

## 🌐 4. Web App (Streamlit)

**File:** `web.py`

Aplikasi ini digunakan untuk:

- 📂 Load model dari folder `new_model/`
- 🧾 Menerima input data dari user
- 🔍 Melakukan prediksi menggunakan model
- 📊 Menampilkan hasil prediksi ke interface web

---

## ▶️ Cara Menjalankan

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Jalankan aplikasi
```bash
streamlit run web.py
```

### 3. Akses di browser
```bash
http://localhost:8501
```
