# Panduan Anotasi NER — Article NER Service
# ==========================================
# Dokumen ini mendefinisikan skema label, aturan batas span,
# dan keputusan desain untuk anotasi Named Entity Recognition.
#
# WAJIB dibaca sebelum memulai anotasi.
# Perubahan harus didiskusikan dan dicatat di annotation_log.md.

## 1. Skema Label

| Label   | Cakupan                                              | Contoh                                      |
|---------|------------------------------------------------------|---------------------------------------------|
| PERSON  | Nama orang (termasuk gelar kehormatan)               | Datuk Seri Ahmad, Tan Sri Ng, John Smith    |
| ORG     | Perusahaan, bank, institusi, partai, LSM, regulator  | Maybank, Bank Negara Malaysia, CIMB, DAP   |
| GPE     | Negara, negara bagian, kota (entitas geopolitik)     | Malaysia, Selangor, Kuala Lumpur           |
| LOC     | Lokasi non-politis                                   | Petronas Twin Towers, Sungai Klang         |
| DATE    | Ekspresi tanggal / periode waktu                     | 7 September 2026, Q3 2026, early August    |
| EVENT   | Acara/peristiwa bernama                              | Maybank Awards 2026, World Economic Forum  |

### Label yang TIDAK dipakai (meski model mungkin menghasilkan):
- MONEY, PERCENT, CARDINAL, ORDINAL, QUANTITY → diabaikan / tidak ditampilkan
- NORP (nationality/religion/political group) → gabungkan ke ORG jika nama lembaga formal, diabaikan jika sifat umum

---

## 2. Keputusan Batas Span

### 2.1 Span tunggal vs split
- **"Bank Negara Malaysia"** → satu ORG (bukan ORG + GPE)
- **"Maybank Berhad"** → satu ORG (termasuk "Berhad")
- **"Tan Sri Dr. Ng Boon Heong"** → satu PERSON (termasuk semua gelar)

### 2.2 Gelar kehormatan Malaysia
Gelar berikut TERMASUK dalam span PERSON:
`Datuk`, `Dato'`, `Tan Sri`, `Tun`, `Encik`, `Puan`, `Dr.`, `Prof.`

Contoh:
- "Datuk Seri Ahmad Zaki" → `[Datuk Seri Ahmad Zaki]_PERSON`
- "Encik Hafiz" → `[Encik Hafiz]_PERSON`

### 2.3 Singkatan organisasi
Singkatan yang berdiri sendiri TETAP dianotasi sebagai ORG jika merujuk ke entitas tertentu:
- "CIMB" → ORG
- "RHB" → ORG
- "BNM" (Bank Negara Malaysia) → ORG
- "DAP" (Democratic Action Party) → ORG

### 2.4 Entitas tumpang tindih / bersarang
Aturan: **tidak ada tumpang tindih**. Pilih span terluar yang bermakna.
- "Maybank Islamic Berhad" → satu ORG (bukan Maybank + Islamic)

### 2.5 Tanda baca di ujung span
- Titik di akhir kalimat TIDAK termasuk span
- Koma setelah nama TIDAK termasuk span
- Tanda kutip yang mengelilingi nama TIDAK termasuk span
- Titik dalam singkatan TERMASUK: "S&P Global" → ORG, "Dr." → bagian PERSON

Contoh:
- "...dihadiri Maybank, CIMB, dan RHB." → tiga span ORG terpisah tanpa titik/koma

### 2.6 Ejaan berbeda untuk entitas yang sama
Anotasi masing-masing sebagai entitas terpisah dengan label yang sama:
- "Maybank" dan "Malayan Banking Berhad" → keduanya ORG, tidak dinormalisasi di anotasi
- Normalisasi bisa dilakukan di lapisan pasca-proses (bukan tugas anotasi)

---

## 3. Panduan per Bahasa

### 3.1 Bahasa Inggris (en)
Ikuti aturan standar di atas.

### 3.2 Bahasa Melayu (ms)
- Nama orang sering diawali gelar → ikuti aturan 2.2
- "Kerajaan Malaysia" → ORG (bukan GPE)
- "Ibu Pejabat" → LOC (jika diikuti nama lokasi), atau diabaikan jika generik

### 3.3 Bahasa Mandarin (zh)
- Anotasi entitas yang bisa diidentifikasi dengan jelas
- Jika ragu, tandai sebagai "uncertain" di annotation_log.md

---

## 4. Kasus Ambigu — Keputusan yang Sudah Dibuat

| Situasi                                    | Keputusan                          |
|--------------------------------------------|------------------------------------|
| "early August 2025"                        | DATE (inklusif untuk ekspresi relatif) |
| "2026" berdiri sendiri (tahun saja)        | DATE                               |
| "Q3 2026"                                  | DATE                               |
| Persentase seperti "5%"                    | BUKAN DATE, abaikan                |
| "Kuala Lumpur" dalam konteks bisnis        | GPE                                |
| "Kuala Lumpur" merujuk kantor/gedung       | LOC                                |
| Nama kemitraan "Maybank-CIMB Partnership" | Satu ORG                           |
| Nama award "Maybank Awards 2026"           | EVENT                              |

---

## 5. Sumber Label

Setiap anotasi harus mencatat sumber label:
- `manual` — dianotasi langsung oleh manusia
- `model-assisted` — disarankan model, **dikoreksi manusia**
- `model-assisted-unreviewed` — TIDAK boleh dipakai untuk evaluasi akhir

---

## 6. Format Anotasi (JSONL)

Setiap baris satu dokumen:

```json
{
  "original_id": "12345",
  "source_file": "Maybank_...hMxhx9QLTq.csv",
  "source_row": 42,
  "title": "Maybank Reports Record Profit",
  "body": "Kuala Lumpur — Maybank announced...",
  "label_source": "model-assisted",
  "reviewed_by": "human",
  "entities": [
    {"text": "Maybank", "label": "ORG", "start": 0, "end": 7},
    {"text": "Kuala Lumpur", "label": "GPE", "start": 16, "end": 28}
  ]
}
```

Offset `start`/`end` dihitung pada `analyzed_text = title + "\n\n" + body`.

---

## 7. Checklist Sebelum Menyimpan Anotasi

- [ ] Semua span dalam batas kata (tidak memotong di tengah kata)
- [ ] Tidak ada span tumpang tindih
- [ ] Label sesuai dengan skema (bukan label output model mentah)
- [ ] Gelar kehormatan sudah dimasukkan dalam span PERSON
- [ ] Tanda baca ujung tidak termasuk dalam span
- [ ] `label_source` terisi dengan benar
- [ ] `reviewed_by` = "human" jika sudah diverifikasi manusia
