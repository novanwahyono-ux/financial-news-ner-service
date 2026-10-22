from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# path data
DATA_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
SPLITS_DIR = ROOT / "data" / "splits"
SPLITS_PATH = SPLITS_DIR / "splits.json"

# parameter proyek
SEED = 42
MAX_INPUT_CHARS = 200000
TITLE_BODY_SEPARATOR = "\n\n"

# buat cek takutnya tabrakan file pake con: Test-Path src\__init__.py, berlaku buat yg lainnya jg

# buat terjemahin nama bank dari original_id
CLIPPING_MAP = {
    "9xtqAsPitG": "Hong Leong Bank",
    "bRQ9afwSIP": "CIMB Group Malaysia",
    "4sxH8Dz0hr": "Public Bank Berhad",
    "58yzyiBQ9U": "RHB Bank",
    "WdjZl0bRfZ": "MDV",
    "i3BJEnT77K": "Maybank CSR",
    "nwub4cROX": "Maybank Foundation",
    "hMxhx9QLTq": "Maybank Group",
}

SPLITS_PATH = SPLITS_DIR / "splits.json"
PROFILING_PATH = PROCESSED_DIR / "profiling_report.json"

# parameter deduplikasi MinHash LSH
NUM_PERM = 128               
SHINGLE_SIZE = 3              
SIMILARITY_THRESHOLD = 0.85    

# besaran perbandingan pembagian data
SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}