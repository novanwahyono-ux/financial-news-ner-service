"""
data_loader.py — Muat dan bersihkan 8 file CSV kliping Maybank
===============================================================
Ketentuan dari brief (Bagian 3.1 & 3.4):
  • Pemisah kolom: titik koma (;)
  • Semua kolom dibaca sebagai string (dtype=str)
  • Awalan kutip tunggal (`'en`, `'ms`) → hapus dengan lstrip("'")
  • Format tanggal: dd/mm/yyyy hh.mm.ss → parse dengan strptime
  • Artikel asli TIDAK di-commit; path diambil dari config.DATA_DIR

Penggunaan:
    from src.data_loader import load_all_csvs
    df = load_all_csvs()
"""

import logging
from pathlib import Path

import pandas as pd

from src.config import (
    CSV_SEPARATOR,
    CLIPPING_MAP,
    DATA_DIR,
    DATE_FORMAT,
    REQUIRED_COLUMNS,
)

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

def _strip_leading_quote(series: pd.Series) -> pd.Series:
    """Hapus awalan kutip tunggal yang salah encode (mis. `'en` → `en`)."""
    return series.str.lstrip("'")


def _parse_dates(series: pd.Series) -> pd.Series:
    """
    Parse tanggal format "dd/mm/yyyy hh.mm.ss" ke datetime.
    Baris yang gagal diparse akan menjadi NaT (tidak crash).
    """
    return pd.to_datetime(series, format=DATE_FORMAT, errors="coerce")


def _clean_dataframe(df: pd.DataFrame, source_file: str) -> pd.DataFrame:
    """
    Bersihkan satu DataFrame hasil read_csv:
    1. Tambahkan kolom `source_file` untuk traceability.
    2. Strip awalan kutip pada kolom teks utama.
    3. Parse tanggal.
    4. Pastikan kolom wajib ada; tambahkan sebagai NaN jika tidak ada.
    """
    df = df.copy()
    df["source_file"] = source_file

    # Pastikan kolom wajib ada
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            logger.warning("Kolom '%s' tidak ditemukan di %s; diisi NaN.", col, source_file)
            df[col] = pd.NA

    # Hapus awalan kutip pada kolom kategorikal / ID
    for col in ("language", "original_id", "source_name"):
        if col in df.columns:
            df[col] = _strip_leading_quote(df[col].fillna(""))
            # Kembalikan string kosong ke NaN supaya konsisten
            df[col] = df[col].replace("", pd.NA)

    df["clipping_name"] = next(
        (name for code, name in CLIPPING_MAP.items() if code in source_file),
        "UNKNOWN",
    )

    # Parse tanggal
    if "date_published" in df.columns:
        df["date_published_parsed"] = _parse_dates(df["date_published"])

    return df


# ──────────────────────────────────────────────────────────────
# Fungsi publik
# ──────────────────────────────────────────────────────────────

def load_csv(path: Path) -> pd.DataFrame:
    """
    Muat satu file CSV dengan aturan:
    - sep=";"
    - dtype=str (semua kolom sebagai teks)
    - quoting standar (csv.QUOTE_MINIMAL) — pandas default sudah benar
    """
    logger.info("Memuat: %s", path.name)
    df = pd.read_csv(
        path,
        sep=CSV_SEPARATOR,
        dtype=str,
        encoding="utf-8",
        on_bad_lines="warn",   # pandas >= 1.3; log baris rusak, tidak crash
    )
    return _clean_dataframe(df, source_file=path.name)


def load_all_csvs(data_dir: Path | None = None) -> pd.DataFrame:
    """
    Muat semua file CSV di `data_dir`, gabungkan, kembalikan satu DataFrame.

    Parameters
    ----------
    data_dir : Path, optional
        Override DATA_DIR dari config jika diberikan.

    Returns
    -------
    pd.DataFrame
        DataFrame gabungan dengan kolom tambahan `source_file`.

    Raises
    ------
    FileNotFoundError
        Jika `data_dir` tidak ada atau tidak ada file CSV di dalamnya.
    """
    dir_path = Path(data_dir) if data_dir else DATA_DIR

    if not dir_path.exists():
        raise FileNotFoundError(
            f"DATA_DIR tidak ditemukan: {dir_path}\n"
            "Atur env var DATA_DIR ke path yang benar."
        )

    csv_files = sorted(dir_path.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"Tidak ada file .csv di {dir_path}")

    logger.info("Ditemukan %d file CSV di %s", len(csv_files), dir_path)

    frames: list[pd.DataFrame] = []
    for csv_path in csv_files:
        try:
            frames.append(load_csv(csv_path))
        except Exception as exc:
            logger.error("Gagal memuat %s: %s", csv_path.name, exc)
            raise

    df_all = pd.concat(frames, ignore_index=True)
    logger.info("Total baris setelah penggabungan: %d", len(df_all))
    return df_all


def load_all(data_dir: Path | None = None) -> pd.DataFrame:
    """Compatibility alias for loading all CSV files."""
    return load_all_csvs(data_dir)


def unique_articles(df: pd.DataFrame) -> pd.DataFrame:
    """Return one row per source article identifier."""
    return df.drop_duplicates(subset=["original_id"]).reset_index(drop=True)
