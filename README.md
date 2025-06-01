# IDX Financial Statement RAG App

  

Sistem **Retrieval-Augmented Generation (RAG)** untuk melakukan *question answering* terhadap laporan keuangan perusahaan yang terdaftar di Bursa Efek Indonesia (IDX).

  

---

  

## 🚀 Fitur

  

- ✅ Ekstraksi otomatis PDF ke Markdown

- ✅ Chunking dengan RecursiveCharacterTextSplitter

- ✅ Embedding hybrid: Dense (MiniLM) + Sparse (BM25)

- ✅ Penyimpanan ke Milvus vector DB

- ✅ Metadata lengkap (aset, laba, sektor, anak perusahaan, dsb)

- ✅ Aplikasi interaktif Streamlit untuk tanya jawab

- ✅ Mendukung model GPT-4o-Latest

  

---

  

## 📁 Struktur Folder

  

```

idx_fin_rag/

├── data/

│ ├── idx/ # PDF asli dari IDX

│ └── processed_text/ # File markdown hasil ekstraksi

├── fin_app.py # Streamlit App

├── process_pdf.py # Ekstrak PDF -> Markdown

├── transform_md.py # ETL Markdown -> Milvus

├── requirements.txt # Dependency Python

├── README.md # Dokumentasi ini

```

  

---

  

## 🛠️ Setup & Instalasi


  

### 1. Install Dependency

  

```bash

pip  install  -r  requirements.txt

```

  

### 2. Ekstrak PDF ke Markdown

  

Masukkan file PDF ke dalam folder `data/idx/` lalu jalankan:

  

```bash

python  process_pdf.py

```

  

### 3. Jalankan Transformasi dan Simpan ke Milvus

  

```bash

python  transform_md.py

```

  

### 4. Jalankan Aplikasi Streamlit

  

```bash

streamlit  run  fin_app.py

```

  

Akses aplikasi di browser: [http://localhost:8501](http://localhost:8501)

  

---

  

## 📌 Contoh Pertanyaan

  

- "Berapa net profit BRI  2024?"

- "Siapa anak perusahaan TLKM?"

- "Berapa total aset  BCA tahun ini??"

  

---

  

## 🔍 Kebutuhan

  

- Python 3.10+

- Milvus 

- Docker

- API Key OpenAI 

  

---


  

## 🔍 Contoh UI
  
```
![plot](/images/BRI_net.png)
```
---

Jika ada pertanyaan atau bug, silakan buka issue di GitHub repo ini 🙌


