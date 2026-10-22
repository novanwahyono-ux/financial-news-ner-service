import csv
from pathlib import Path
import pandas as pd
from src.config import CLIPPING_MAP, DATA_DIR

DATE_FORMAT = "%d/%m/%Y %H.%M.%S"

# bersihin nama kolom yg depannya ada '
CLEAN_COLUMNS = ("language", "original_id", "source_name")

def _strip_leading_quote(series):
    return series.str.lstrip("'")

def _parse_date(series):
    return pd.to_datetime(series, format=DATE_FORMAT, errors="coerce")

def _clipping_name(filename):
    for code, name in CLIPPING_MAP.items():
        if code in filename:
            return name
    return "UNKNOWN"

def load_csv(filepath):
    filepath = Path(filepath)
    df = pd.read_csv(filepath, sep=";", dtype=str, quoting=csv.QUOTE_MINIMAL)
    for col in CLEAN_COLUMNS:
        df[col] = _strip_leading_quote(df[col])
    df["date_published"] = _parse_date(df["date_published"])
    df["source_file"] = filepath.name
    df["clipping_name"] = _clipping_name(filepath.name)
    return df

def load_all(data_dir=None):
    data_dir = Path(data_dir) if data_dir else DATA_DIR
    files = sorted(data_dir.glob("*.csv")) #ngumpulin file format .csv
    if not files:
        raise FileNotFoundError(f"Tidak ada file .csv di {data_dir}")
    return pd.concat([load_csv(f) for f in files], ignore_index=True)

def unique_articles(df):
    return df.drop_duplicates(subset="original_id").reset_index(drop=True) #ini buat split data kembar (takutna duplikat)