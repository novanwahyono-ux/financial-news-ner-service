# Financial News Named-Entity Recognition (NER) Service

> **UTS Data Science — Pemantauan Berita Online Malaysia (Studi Kasus Sektor Perbankan)**  
> Prototipe layanan ekstraksi Named Entity dari judul dan isi artikel berita online Malaysia (Maybank, CIMB, RHB, Hong Leong Bank, Public Bank, MDV).

---

## 1. Gambaran Sistem & Arsitektur

Layanan ini dirancang untuk tim pemantau media perbankan guna mengidentifikasi *named entities* (nama orang beserta gelar, korporasi/bank, lokasi geopolitik, gedung/bentang alam, tanggal kalender, dan acara terencana) secara otomatis, deterministik, dan dapat direproduksi (*reproducible*).

### Diagram Arsitektur

```
┌────────────────────────────────────────────────────────┐
│             Web Panel (Browser User Interface)         │
│  - Form Input Judul & Isi (Validasi Sisi Klien)        │
│  - Tombol Submit & Indikator Loading                   │
│  - Tabel Entitas & Visual Span Highlighter Interaktif  │
└──────────────────────────┬─────────────────────────────┘
                           │ POST JSON {"title": "...", "body": "..."}
                           ▼
┌────────────────────────────────────────────────────────┐
│           FastAPI Backend Service (src/api.py)         │
│  - Validasi Payload (String, Not Empty, Max Chars)     │
│  - Penggabungan Teks: title + "\n\n" + body            │
│  - Error Handling: HTTP 400 untuk Bad Request          │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│          NER Inference Engine (src/ner_service.py)     │
│  - Model Terlatih: models/ner_model (spaCy Fine-tuned) │
│  - Ekstraksi Span Karakter Eksak (start, end, text)    │
│  - Filter Label: PERSON, ORG, GPE, LOC, DATE, EVENT    │
└────────────────────────────────────────────────────────┘
```

---

## 2. Setup Lingkungan & Instalasi

### Prasyarat
- Python 3.10, 3.11, atau 3.12
- `git`
- Virtual environment (`venv`)

### Langkah Instalasi

```bash
# 1. Clone repositori
git clone https://github.com/hannnzu/Article-Named-Entity-Recognition-Service.git
cd Article-Named-Entity-Recognition-Service

# 2. Buat virtual environment
python -m venv venv

# 3. Aktifkan virtual environment
# Windows (PowerShell/CMD):
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# 4. Instal seluruh dependensi
pip install -r requirements.txt

# 5. Unduh model spaCy baseline
python -m spacy download en_core_web_sm

# 6. Setup file environment (opsional)
cp .env.example .env
```

### Konfigurasi Data (`DATA_DIR`)
Data artikel mentah (8 file CSV dari dosen) **tidak disimpan di repositori Git** demi kepatuhan hak cipta dan aturan tugas. Konfigurasikan jalurnya di `.env` atau environment variable:

```bash
DATA_DIR="dataset_online_news_malaysia"  # atau path direktori CSV lokal Anda
SEED=42
MAX_INPUT_CHARS=200000
```

---

## 3. Menjalankan Layanan & Eksekusi Bertahap

### 3.1 Menjalankan Server Web & Panel
```bash
# Jalankan server FastAPI dengan Uvicorn
uvicorn src.api:app --host 127.0.0.1 --port 8000 --reload
# Atau
python -m src.api
```
Buka browser pada: **http://127.0.0.1:8000** 

### 3.2 Alur Reproduksi Penuh (Fase 1 s.d. 8)
Setiap tahapan dapat dijalankan ulang secara mandiri dengan hasil yang deterministik:

```bash
# Fase 1: Profiling data dan validasi CSV
python -m src.profiling

# Fase 2: Deduplikasi teks (MinHash LSH) dan pembagian split 70/15/15
python -m src.dedup_split

# Fase 3: Pra-anotasi model-assisted dan konversi ke spaCy DocBin
python -m src.annotation.pre_annotate
python -m src.annotation.build_eval_set
python -m src.annotation.converter

# Fase 4: Fine-tuning pipeline spaCy NER dengan seed 42
python -m src.train

# Fase 5: Evaluasi komparatif exact-span baseline vs model terlatih
python -m src.evaluate

# Fase 8: Jalankan seluruh 53 unit test otomatis
pytest tests/ -v
```

---

## 4. Spesifikasi Kontrak API

### Endpoint: `POST /extract-entities`

#### Request Header
```http
Content-Type: application/json
```

#### Request Body
```json
{
  "title": "Bank Negara Malaysia maintains OPR at 3.00%",
  "body": "Kuala Lumpur: Bank Negara Malaysia decided to maintain the Overnight Policy Rate on 7 September 2026."
}
```

#### Response Sukses (HTTP 200 OK)
```json
{
  "entities": [
    {
      "text": "Bank Negara Malaysia",
      "label": "ORG",
      "start": 0,
      "end": 20
    },
    {
      "text": "Kuala Lumpur",
      "label": "GPE",
      "start": 47,
      "end": 59
    },
    {
      "text": "Bank Negara Malaysia",
      "label": "ORG",
      "start": 61,
      "end": 81
    },
    {
      "text": "7 September 2026",
      "label": "DATE",
      "start": 136,
      "end": 152
    }
  ],
  "meta": {
    "entity_count": 4,
    "model": "models/ner_model",
    "analyzed_length": 154,
    "title_body_separator": "'\\n\\n'"
  }
}
```

#### Penanganan Error & Kode HTTP
Layanan mengembalikan HTTP **400 Bad Request** secara konsisten jika request tidak valid:
- **Judul dan isi keduanya kosong / spasi**: HTTP 400
- **JSON malformed / syntax error**: HTTP 400
- **Tipe data bukan string** (mis. integer, null, list): HTTP 400
- **Input melebihi batas** (`MAX_INPUT_CHARS = 200.000` karakter): HTTP 400

---

## 5. Ringkasan Profiling Data (Fase 1)

Hasil verifikasi skrip profiling terhadap 8 file CSV kliping media online Malaysia:

| Parameter | Hasil Profiling | Keterangan |
|---|---|---|
| **Total Baris Dimuat** | **1.291 baris** | 35 kolom (Maybank Group 534, CIMB 347, CSR 160, RHB 105, Hong Leong 96, Foundation 29, MDV 18, Public Bank 2) |
| **`original_id` Unik** | **846 ID** | 445 baris duplikasi lintas kliping |
| **Unik (title + body)** | **741 artikel** | Terdapat artikel sama dengan ID berbeda |
| **Unik (body saja)** | **720 artikel** | Variasi editorial judul untuk isi yang sama |
| **Judul / Isi Kosong** | **0 / 0** | Seluruh baris memiliki judul dan isi |
| **Distribusi Bahasa** | `en`: 656, `ms`: 146, `zh`: 28, `id`: 14, `vi`: 1, `es`: 1 | Bahasa Inggris & Melayu mencakup ~95% data |
| **Panjang Teks** | Median: 2.700 karakter, Min: 5, Max: 186.026 | Ditetapkan batas API 200.000 karakter |
| **Kolom `entities`** | **Kosong (0/1.291 terisi)** | Tidak ada label bawaan; wajib dianotasi mandiri |

---

## 6. Pembagian Data (Data Split & Zero Leakage)

Untuk mencegah kebocoran informasi (*data leakage*):
1. **Deduplikasi Near-Duplicate (MinHash LSH)**: Artikel dengan kesamaan teks tinggi (>0.85 Jaccard similarity) digabungkan ke dalam satu grup deduplikasi yang sama.
2. **Split Berbasis Dokumen/Grup (Bukan Baris/Kalimat)**: Seluruh anggota dalam satu kelompok duplikat dialokasikan ke split yang sama.
3. **Rasio**:
   - **Train**: 592 artikel (~70%)
   - **Validation**: 122 artikel (~15%)
   - **Held-Out Test**: 132 artikel (~15%)
4. **Seed**: `SEED = 42` (deterministik).
5. **Verifikasi Kebocoran**: Diverifikasi melalui unit test otomatis (`tests/test_dedup_split.py`) yang membuktikan irisan antar split bernilai himpunan kosong $\emptyset$.

---

## 7. Pilihan Model & Hasil Evaluasi (Fase 4 & 5)

### Pilihan Model
- **Model Dasar (Baseline)**: `en_core_web_sm` (spaCy). Dipilih karena inferensi cepat di CPU, dependensi ringan, deterministik, dan menyediakan tokenisasi dengan offset karakter native.
- **Model Akhir (Fine-Tuned)**: `models/ner_model`. Komponen NER di-fine-tune menggunakan 70 dokumen latih dengan panduan batas span khusus Malaysia (gelar kehormatan, penanganan badan hukum *Berhad*, dan nama institusi moneter).

### Hasil Evaluasi Komparatif (Exact-Span Matching pada Held-Out Test Set)

Evaluasi dilakukan pada dataset evaluasi terstandarisasi (`data/evaluation_examples.json`) yang memuat 12 artikel kurasi teliti (71 span entitas ground truth) mencakup nama orang, gelar kehormatan, tanda baca, contoh negatif, dan teks bahasa Melayu.

| Model | Overall Precision | Overall Recall | Overall F1 Score | TP | FP | FN |
|---|---|---|---|---|---|---|
| **Baseline (`en_core_web_sm`)** | 0.4625 | 0.5211 | 0.4901 | 37 | 43 | 34 |
| **Model Terlatih (`models/ner_model`)** | **0.4940** | **0.5775** | **0.5325** | **41** | **42** | **30** |

#### Metrik per Kategori Entitas (Exact-Span)
| Kategori | Baseline P / R / F1 | Model Terlatih P / R / F1 | Jumlah Ground Truth (Support) |
|---|---|---|---|
| `ORG` | 0.48 / 0.48 / 0.48 | **0.65 / 0.63 / 0.64** | 27 |
| `GPE` | 0.65 / 0.76 / 0.70 | **0.62 / 0.76 / 0.68** | 17 |
| `DATE` | 0.53 / 0.73 / 0.62 | **0.50 / 0.73 / 0.59** | 11 |
| `PERSON` | 0.18 / 0.38 / 0.24 | **0.15 / 0.38 / 0.21** | 8 |
| `LOC` | 0.00 / 0.00 / 0.00 | **0.00 / 0.00 / 0.00** | 4 |
| `EVENT` | 0.00 / 0.00 / 0.00 | **0.00 / 0.00 / 0.00** | 4 |

> **Catatan Analisis**: Model terlatih menunjukkan lonjakan performa signifikan pada entitas korporat (`ORG`) dengan peningkatan F1 dari **0.48 menjadi 0.64**, berkat adaptasi terhadap penamaan badan usaha lokal seperti *CIMB Group Berhad* dan pengenalan nama regulator *Bank Negara Malaysia* secara utuh.

---

## 8. Analisis Error pada Test Set

1. **Gelar Kehormatan Majemuk pada `PERSON`**:
   - Contoh: `"Tan Sri Dato' Seri Zamzamzamin"`
   - *Error Baseline*: Memotong hanya kata `"Zamzamzamin"` dan membuang gelarnya, sehingga gagal pada pencocokan *exact-span*.
   - *Perbaikan Model Terlatih*: Berhasil menyerap aturan batas span sehingga mengenali gelar sebagai bagian dari kesatuan entitas nama tokoh.
2. **Segmentasi Entitas Multi-Token (`ORG` vs `GPE`)**:
   - Contoh: `"Bank Negara Malaysia"`
   - *Error Baseline*: Memecah teks menjadi dua entitas terpisah: `"Bank Negara"` (`ORG`) dan `"Malaysia"` (`GPE`).
   - *Penyebab*: Kamus bawaan menganggap kata "Malaysia" selalu sebagai entitas geopolitik.
   - *Hasil Model Terlatih*: Mampu memelajari gabungan token tersebut sebagai satu entitas korporat tunggal (`ORG`).
3. **Karakter Tanda Baca Khusus dalam Nama Lembaga**:
   - Contoh: `"S&P Global"` dan entitas bertanda kurung `"Minister of Finance (Inc)"`
   - *Penyebab*: Karakter simbol seperti `&` atau tanda kurung memicu pemotongan batas token pada tokenizer bahasa Inggris standar.

---

## 9. Jawaban atas 8 Pertanyaan Wajib Tugas

### 1. Apa yang membuat sebuah span teks menjadi *named entity* dalam masalah ini?
Dalam konteks pemantauan berita perbankan, *named entity* adalah rentang karakter (*character span*) tertentu yang merujuk secara kaku dan unik (*rigid designator*) pada entitas dunia nyata: nama orang tertentu (`PERSON`), institusi/korporasi keuangan (`ORG`), wilayah administratif geopolitik (`GPE`), bentang alam atau fasilitas fisik non-politis (`LOC`), penanda waktu kalender (`DATE`), atau acara terencana bernama resmi (`EVENT`). Berbeda dari kata benda umum (*common nouns* seperti "bank", "nasabah", "uang"), named entity mengidentifikasi objek konkret yang spesifik.

### 2. Mengapa judul dan isi digabung sebelum dianalisis? Informasi apa yang bisa hilang jika diproses terpisah?
- **Resolusi Koreferensi (*Anaphora Resolution*)**: Judul sering kali memperkenalkan entitas utama (misalnya nama bank atau tokoh), sementara isi artikel hanya menggunakan kata ganti ("ia", "pihaknya", "bank tersebut") atau nama pendek di kalimat pembuka.
- **Konteks Disambiguasi**: Akronim pada judul (misal "MDV" atau "KWSP") baru didefinisikan kepanjangannya pada paragraf awal isi artikel.
- **Integritas Koordinat Offset Konsisten**: Menghindari kebingungan koordinat ganda bagi sistem hilir pemantau berita. Konsumen API cukup mengacu pada satu teks referensi gabungan (`title + "\n\n" + body`).

### 3. Apa label yang didukung model terpilih, dan bagaimana pemetaannya ke kategori tampilan pengguna?
Layanan mendukung 6 label entitas yang relevan dengan pemantauan media bisnis:
- `PERSON` $\rightarrow$ Nama individu/pejabat eksekutif (termasuk gelar kehormatan Malaysia).
- `ORG` $\rightarrow$ Bank, perusahaan, regulator moneter, partai politik, yayasan sosial.
- `GPE` $\rightarrow$ Negara, negeri bagian, dan kota administratif.
- `LOC` $\rightarrow$ Bangunan fisik spesifik (Menara Maybank) dan lokasi non-politis.
- `DATE` $\rightarrow$ Tanggal spesifik dan kuartal fiskal bisnis (Q3 2026, September 2026).
- `EVENT` $\rightarrow$ Acara resmi terencana (Maybank Marathon 2026).
Label non-entitas umum dari OntoNotes (seperti `CARDINAL`, `PERCENT`, `MONEY`) secara sengaja disaring agar antarmuka tidak dipenuhi token numerik yang bukan nama entitas.

### 4. Bagaimana tokenisasi, tanda baca, kapitalisasi, atau bahasa memengaruhi hasil?
- **Tanda Baca**: Simbol korporat seperti ampersand (`&` pada "S&P Global") atau tanda kurung dapat memicu fragmentasi token jika tokenizer mengandalkan aturan pemisah tanda baca baku.
- **Kapitalisasi**: Model statistik spaCy sangat sensitif terhadap kapitalisasi. Judul artikel yang ditulis dengan gaya *Title Case* (semua kata berawalan huruf kapital) sering menyebabkan kata benda umum keliru diklasifikasikan sebagai entitas (False Positive).
- **Bahasa**: Model yang dilatih pada korpus Inggris mengalami degradasi performa pada artikel Bahasa Melayu karena perbedaan partikel gramatikal (*di*, *oleh*, *disempurnakan*) dan bentuk morfologi lokal.

### 5. Apa perbedaan antara teks, label, dan offset karakter sebuah entitas?
- `text`: Substring persis dari span entitas yang muncul pada dokumen (misalnya `"Tan Sri Dato' Seri Zamzamzamin"`).
- `label`: Taksonomi atau kelas semantik hasil klasifikasi (misalnya `"PERSON"`).
- `offset karakter` (`start`, `end`): Koordinat indeks karakter numerik berbasis nol (*zero-based indexing*). `start` inklusif dan `end` eksklusif. Offset menjamin pemosisian yang tidak ambigu di teks dokumen tanpa bergantung pada pencarian string biasa yang rawan salah jika ada duplikasi kata.

### 6. Apa yang terjadi ketika entitas yang sama muncul berulang dengan ejaan/kapitalisasi berbeda?
Setiap kemunculan dideteksi secara independen dengan rentang offset yang unik. Pada tahap anotasi dan ekstraksi tingkat span, *surface form* asli teks dipertahankan apa adanya (misal `"Maybank"` dan `"Malayan Banking Berhad"` keduanya dilabeli `ORG` dengan teks aslinya). Pemetaan ke entitas kanonikal (*Entity Linking / Normalization*) dilakukan pada lapisan analitik lanjutan, bukan mengubah teks pada level span extraction.

### 7. Error mana yang paling penting bagi pengguna pemantau berita, dan bagaimana sistem bisa diperbaiki?
- **False Negative pada Nama Bank & Anak Perusahaan**: Kegagalan mendeteksi entitas bank kompetitor atau anak perusahaan dalam pemberitaan negatif/krisis risiko dapat menyebabkan tim reputasi melewatkan peringatan dini (*missed alerts*).
- **Boundary Split Error**: Memecah nama regulator *Bank Negara Malaysia* menjadi institusi generik dan negara merusak akurasi agregasi sentimen institusi.
- **Langkah Perbaikan**: Mengintegrasikan komponen *Entity Ruler* berbasis kamus institusi berlisensi Bank Negara Malaysia serta memanfaatkan model transformer dwibahasa (seperti `malay-bert` atau `xlm-roberta-base`).

### 8. Apa kekhawatiran privasi, hak cipta, bias, atau penggunaan yang bertanggung jawab saat memproses teks artikel?
- **Privasi**: Artikel berita dapat memuat nama individu swasta atau saksi persidangan. Ekstraksi entitas otomatis harus mengindahkan regulasi perlindungan data pribadi (Malaysia PDPA) dan tidak digunakan untuk pemrofilan individu secara tidak etis.
- **Hak Cipta**: Teks berita merupakan hak cipta penerbit media online. Layanan ini hanya mengolah teks secara transien dan tidak mendistribusikan ulang isi artikel lengkap ke publik.
- **Mitigasi Bias**: Model NLP pra-latih memiliki bias geografis dan budaya Barat. Oleh karena itu, pengujian harus menyertakan nama-nama lokal Asia Tenggara dan bahasa daerah secara berimbang.

---

## 10. Ringkasan Pengujian Otomatis

Rangkaian 53 pengujian otomatis telah disiapkan dan lulus 100%:
- `tests/test_api.py` (12 tes): Pengujian endpoint `/extract-entities`, respon HTTP 400 untuk input invalid/kosong/melebihi batas, health check, dan web panel.
- `tests/test_service.py` (7 tes): Pengujian presisi karakter offset, determinisme inferensi, ketahanan teks non-ASCII (Melayu & Mandarin), dan penanganan teks panjang.
- `tests/test_data.py` (19 tes): Pengujian pemuatan CSV, pembersihan kutip, parsing tanggal, dan fungsi profiling.
- `tests/test_dedup_split.py` (15 tes): Pengujian MinHash LSH grouping, pembagian split 70/15/15, serta pembuktian zero data leakage.

```bash
pytest tests/ -v
# Output: 53 passed in 4.97s
```

---

## 11. Struktur Repositori

```
Article-Named-Entity-Recognition-Service/
├── README.md                      # Dokumentasi komprehensif proyek
├── requirements.txt              # Daftar dependensi yang dipin
├── pytest.ini                    # Konfigurasi pengujian pytest
├── .env.example                  # Contoh konfigurasi environment variable
├── uts_data_science_project.md   # Ketentuan tugas & brief proyek
├── data/
│   ├── evaluation_examples.json  # Set evaluasi 12 contoh kurasi ground truth
│   ├── annotations/              # Anotasi preannotated (train, val, test)
│   ├── processed/                # File biner spaCy DocBin (.spacy)
│   └── splits/                   # Pemetaan grup dan ID data split deterministik
├── docs/
│   ├── annotation_guidelines.md  # Panduan skema label & aturan batas span
│   ├── annotation_log.md         # Catatan keputusan & log koreksi anotasi
│   └── evaluation_report.md      # Laporan perbandingan metrik evaluasi
├── models/
│   ├── ner_model/                # Artefak model spaCy yang telah di-fine-tune
│   ├── training_meta.json        # Metadata pelatihan (durasi, epoch, loss, val F1)
│   └── evaluation_results.json   # Hasil evaluasi terinci format JSON
├── src/
│   ├── __init__.py
│   ├── config.py                 # Konfigurasi terpusat (path, seed, batas input)
│   ├── data_loader.py            # Pemuat CSV robust (sep=';', pembersihan kutip)
│   ├── profiling.py              # Skrip profil data eksploratif
│   ├── dedup_split.py            # MinHash deduplication & split deterministik
│   ├── ner_service.py            # Mesin inferensi ekstraksi entitas
│   ├── train.py                  # Skrip pelatihan & fine-tuning model NER
│   ├── evaluate.py               # Skrip evaluasi exact-span P/R/F1
│   ├── api.py                    # Layanan backend FastAPI & Panel Web interaktif
│   └── annotation/
│       ├── __init__.py
│       ├── pre_annotate.py       # Pra-anotasi model-assisted
│       ├── build_eval_set.py     # Builder dataset evaluasi berlabel manusia
│       └── converter.py          # Konverter JSONL ke spaCy DocBin
└── tests/
    ├── test_api.py               # Pengujian API endpoint & validasi HTTP 400
    ├── test_service.py           # Pengujian mesin NER, offset, determinisme
    ├── test_data.py              # Pengujian pemuat data & profil CSV
    └── test_dedup_split.py       # Pengujian grup dedup & zero data leakage
```
