"""
config.py — Konfigurasi terpusat Article NER Service
=====================================================
Semua konstanta, path, dan hyperparameter dikelola di sini.
Nilai default dapat di-override lewat environment variable atau file .env.

Cara pakai:
    from src.config import settings
    print(settings.DATA_DIR)
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Muat .env jika ada (tidak wajib)
load_dotenv()

# ──────────────────────────────────────────────────────────────
# Direktori root proyek (satu level di atas src/)
# ──────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent
ROOT = ROOT_DIR

# ──────────────────────────────────────────────────────────────
# Path data
# DATA_DIR  : direktori yang berisi 8 file CSV dari dosen.
#             Atur lewat env var DATA_DIR.  Tidak perlu berada
#             di dalam repo; artikel asli TIDAK di-commit.
# ──────────────────────────────────────────────────────────────
DATA_DIR: Path = Path(
    os.getenv(
        "DATA_DIR",
        str(ROOT_DIR / "dataset_online_news_malaysia"),   # default lokal
    )
)

# Direktori output yang aman (di dalam repo, di-gitignore jika perlu)
DATA_PROCESSED_DIR: Path = ROOT_DIR / "data" / "processed"
DATA_SPLITS_DIR: Path = ROOT_DIR / "data" / "splits"
PROCESSED_DIR = DATA_PROCESSED_DIR
SPLITS_DIR = DATA_SPLITS_DIR
SPLITS_PATH: Path = DATA_SPLITS_DIR / "splits.json"
PROFILING_PATH: Path = DATA_PROCESSED_DIR / "profiling_report.json"
EVALUATION_FILE: Path = ROOT_DIR / "data" / "evaluation_examples.json"
ANNOTATION_DIR: Path = ROOT_DIR / "data" / "annotations"
MODELS_DIR: Path = ROOT_DIR / "models"

# ──────────────────────────────────────────────────────────────
# Reprodusibilitas
# ──────────────────────────────────────────────────────────────
SEED: int = int(os.getenv("SEED", "42"))

# ──────────────────────────────────────────────────────────────
# Validasi input API
# MAX_INPUT_CHARS : panjang maksimum GABUNGAN title + "\n\n" + body
#                   dalam karakter.  Artikel di atas batas ini ditolak
#                   dengan HTTP 400.  Didokumentasikan di README.
# ──────────────────────────────────────────────────────────────
MAX_INPUT_CHARS: int = int(os.getenv("MAX_INPUT_CHARS", "200_000"))

# Format penggabungan title + body (harus konsisten di semua modul)
# Offset start/end entitas dihitung pada teks gabungan ini.
TITLE_BODY_SEPARATOR: str = "\n\n"

# ──────────────────────────────────────────────────────────────
# Model NER
# MODEL_NAME : nama model spaCy yang dipakai.
#   baseline  : "en_core_web_sm"
#   multibahasa: "xx_ent_wiki_sm"
#   fine-tuned: path ke model tersimpan, mis. "models/ner_finetuned"
# ──────────────────────────────────────────────────────────────
_default_model = str(MODELS_DIR / "ner_model") if (MODELS_DIR / "ner_model").exists() else "en_core_web_sm"
MODEL_NAME: str = os.getenv("MODEL_NAME", _default_model)

# ──────────────────────────────────────────────────────────────
# Data split
# ──────────────────────────────────────────────────────────────
TRAIN_RATIO: float = 0.70
VAL_RATIO: float = 0.15
TEST_RATIO: float = 0.15   # harus = 1 - TRAIN - VAL
SPLIT_RATIOS = {
    "train": TRAIN_RATIO,
    "val": VAL_RATIO,
    "test": TEST_RATIO,
}

# Parameter MinHash LSH untuk deduplikasi artikel
NUM_PERM: int = 128
SHINGLE_SIZE: int = 3
SIMILARITY_THRESHOLD: float = 0.85

CLIPPING_MAP: dict[str, str] = {
    "9xtqAsPitG": "Hong Leong Bank",
    "bRQ9afwSIP": "CIMB Group Malaysia",
    "4sxH8Dz0hr": "Public Bank Berhad",
    "58yzyiBQ9U": "RHB Bank",
    "WdjZl0bRfZ": "MDV",
    "i3BJEnT77K": "Maybank CSR",
    "nwub4cROX": "Maybank Foundation",
    "hMxhx9QLTq": "Maybank Group",
}

# ──────────────────────────────────────────────────────────────
# Kolom CSV yang dipakai
# ──────────────────────────────────────────────────────────────
CSV_SEPARATOR: str = ";"
CSV_DTYPE: str = "str"      # baca semua kolom sebagai string untuk keamanan

REQUIRED_COLUMNS: list[str] = [
    "title",
    "body",
    "language",
    "entities",
    "original_id",
    "source_name",
    "date_published",
]

# Format tanggal di CSV: "dd/mm/yyyy hh.mm.ss" (pemisah jam = titik)
DATE_FORMAT: str = "%d/%m/%Y %H.%M.%S"

# ──────────────────────────────────────────────────────────────
# Label NER yang didukung layanan (subset dari output model)
# Skema lengkap ada di docs/annotation_guidelines.md
# ──────────────────────────────────────────────────────────────
SUPPORTED_LABELS: list[str] = ["PERSON", "ORG", "GPE", "LOC", "DATE", "EVENT"]

# ──────────────────────────────────────────────────────────────
# Pastikan direktori output ada saat modul diimpor
# ──────────────────────────────────────────────────────────────
for _dir in (DATA_PROCESSED_DIR, DATA_SPLITS_DIR, ANNOTATION_DIR, MODELS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)
