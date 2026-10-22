"""
profiling.py — Profiling dataset 8 CSV kliping Maybank
=======================================================
Sesuai Fase 1 brief (Bagian 3.2 & 3.3), skrip ini melaporkan:
  1. Jumlah baris dimuat (total & per kliping / source_file)
  2. Duplikat: berdasarkan original_id, title+body, body saja
  3. Judul / isi kosong
  4. Distribusi bahasa (setelah dedup original_id)
  5. Statistik panjang teks (body)
  6. Status kolom `entities`

Angka-angka diverifikasi terhadap tabel di Bagian 3.3. Jika berbeda,
skripnya yang dipercaya dan perbedaannya dicatat.

Penggunaan:
    python -m src.profiling            # output ke stdout + simpan JSON
    python -m src.profiling --help
"""

import json
import logging
import sys
from pathlib import Path

import pandas as pd

from src.config import DATA_PROCESSED_DIR
from src.data_loader import load_all_csvs

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

# Mapping suffix file → nama kliping (Bagian 3.2)
CLIPPING_MAP: dict[str, str] = {
    "9xtqAsPitG": "Hong Leong Bank",
    "bRQ9afwSIP": "CIMB Group Malaysia",
    "4sxH8Dz0hr": "Public Bank Berhad",
    "58yzyiBQ9U": "RHB Bank",
    "WdjZl0bRfZ": "MDV",
    "i3BJEnT77K": "Maybank CSR",
    "-nwub4cROX": "Maybank Foundation",
    "hMxhx9QLTq": "Maybank Group",
}

# Angka referensi dari brief Bagian 3.3 (untuk verifikasi)
REFERENCE = {
    "total_rows": 1291,
    "unique_original_id": 846,
    "unique_title_body": 741,
    "unique_body": 720,
    "empty_title": 0,
    "empty_body": 0,
}


# ──────────────────────────────────────────────────────────────
# Helper: resolusi nama kliping dari nama file
# ──────────────────────────────────────────────────────────────

def _clipping_name(filename: str) -> str:
    for suffix, name in CLIPPING_MAP.items():
        if suffix in filename:
            return name
    return filename  # fallback: tampilkan nama file asli


def _normalize_text(s: pd.Series) -> pd.Series:
    """Lowercase + strip whitespace untuk keperluan dedup."""
    return s.fillna("").str.lower().str.strip()


# ──────────────────────────────────────────────────────────────
# Fungsi profiling
# ──────────────────────────────────────────────────────────────

def profile(df: pd.DataFrame) -> dict:
    """
    Hitung semua statistik profiling dan kembalikan sebagai dict.
    Juga cetak laporan ke stdout.
    """
    report: dict = {}

    # ── 1. Jumlah baris ───────────────────────────────────────
    total_rows = len(df)
    report["total_rows"] = total_rows

    # Baris per kliping
    df["clipping_name"] = df["source_file"].apply(_clipping_name)
    rows_per_clipping = (
        df.groupby("clipping_name").size().sort_values(ascending=False).to_dict()
    )
    report["rows_per_clipping"] = rows_per_clipping

    # ── 2. Duplikat ───────────────────────────────────────────
    dup_by_orig_id = total_rows - df["original_id"].nunique()
    report["duplicates_by_original_id"] = dup_by_orig_id
    report["duplicate_rows"] = dup_by_orig_id
    report["unique_original_id"] = df["original_id"].nunique()

    # Dedup title+body (normalisasi sebelum cek)
    df["_key_title_body"] = (
        _normalize_text(df["title"]) + "|||" + _normalize_text(df["body"])
    )
    unique_title_body = df["_key_title_body"].nunique()
    report["unique_title_body"] = unique_title_body
    report["duplicates_by_title_body"] = total_rows - unique_title_body

    # Dedup body saja
    df["_key_body"] = _normalize_text(df["body"])
    unique_body = df["_key_body"].nunique()
    report["unique_body"] = unique_body
    report["duplicates_by_body"] = total_rows - unique_body

    # ── 3. Judul / isi kosong ─────────────────────────────────
    empty_title = df["title"].fillna("").str.strip().eq("").sum()
    empty_body = df["body"].fillna("").str.strip().eq("").sum()
    report["empty_title"] = int(empty_title)
    report["empty_body"] = int(empty_body)
    report["missing_title"] = int(empty_title)
    report["missing_body"] = int(empty_body)

    # ── 4. Distribusi bahasa (setelah dedup original_id) ──────
    df_dedup_id = df.drop_duplicates(subset=["original_id"])
    lang_dist = (
        df_dedup_id["language"].fillna("unknown").value_counts().to_dict()
    )
    report["language_distribution_after_dedup_original_id"] = lang_dist
    report["language_unique_articles"] = lang_dist

    # ── 5. Statistik panjang teks (body) ─────────────────────
    body_len = df["body"].fillna("").str.len()
    report["body_length_stats"] = {
        "count": int(body_len.count()),
        "min": int(body_len.min()),
        "max": int(body_len.max()),
        "mean": round(float(body_len.mean()), 1),
        "median": round(float(body_len.median()), 1),
        "p25": round(float(body_len.quantile(0.25)), 1),
        "p75": round(float(body_len.quantile(0.75)), 1),
        "articles_lt_200_chars": int((body_len < 200).sum()),
        "articles_gt_20000_chars": int((body_len > 20_000).sum()),
    }

    # ── 6. Status kolom `entities` ────────────────────────────
    entities_col = df["entities"].fillna("")
    entities_non_empty = entities_col.str.strip().ne("").sum()
    report["entities_column"] = {
        "total_rows": total_rows,
        "non_empty": int(entities_non_empty),
        "empty": int(total_rows - entities_non_empty),
        "note": (
            "Kolom 'entities' kosong di semua baris -> "
            "tidak tersedia label ground truth dari sumber data. "
            "Tidak dipakai sebagai supervisi tanpa anotasi manual."
        ),
    }
    report["entities_empty_pct"] = (
        round(float((total_rows - entities_non_empty) / total_rows * 100), 2)
        if total_rows
        else 0.0
    )

    dates = df.get("date_published_parsed")
    if dates is None:
        dates = (
            pd.to_datetime(df["date_published"], errors="coerce")
            if "date_published" in df
            else pd.Series(pd.NaT, index=df.index)
        )
    report["date_min"] = str(dates.min()) if dates.notna().any() else "NaT"
    report["date_max"] = str(dates.max()) if dates.notna().any() else "NaT"
    report["unparsed_dates"] = int(dates.isna().sum())

    # ── 7. Verifikasi terhadap referensi brief ────────────────
    deviations: dict = {}
    checks = {
        "total_rows": report["total_rows"],
        "unique_original_id": report["unique_original_id"],
        "unique_title_body": report["unique_title_body"],
        "unique_body": report["unique_body"],
        "empty_title": report["empty_title"],
        "empty_body": report["empty_body"],
    }
    for key, actual in checks.items():
        expected = REFERENCE.get(key)
        if expected is not None and actual != expected:
            deviations[key] = {"expected": expected, "actual": actual, "diff": actual - expected}
    report["reference_check"] = {
        "deviations": deviations,
        "status": "OK — semua sesuai referensi" if not deviations else f"PERHATIAN — {len(deviations)} perbedaan",
    }

    # ── Print laporan ─────────────────────────────────────────
    _print_report(report, df)

    return report


def _print_report(report: dict, df: pd.DataFrame) -> None:
    sep = "=" * 60
    print(f"\n{sep}")
    print("  LAPORAN PROFILING DATASET — Article NER Service")
    print(sep)

    print(f"\n[STAT] TOTAL BARIS      : {report['total_rows']:,}")
    print("\n[KLIPING] BARIS PER KLIPING:")
    for clipping, count in report["rows_per_clipping"].items():
        print(f"   {clipping:<30} {count:>5}")

    print("\n[DEDUP] DUPLIKAT:")
    print(f"   Unik berdasarkan original_id  : {report['unique_original_id']:,}  ({report['duplicates_by_original_id']:,} duplikat)")
    print(f"   Unik berdasarkan title+body   : {report['unique_title_body']:,}  ({report['duplicates_by_title_body']:,} duplikat)")
    print(f"   Unik berdasarkan body         : {report['unique_body']:,}  ({report['duplicates_by_body']:,} duplikat)")

    print("\n[KOSONG] JUDUL / ISI KOSONG:")
    print(f"   Judul kosong : {report['empty_title']}")
    print(f"   Isi kosong   : {report['empty_body']}")

    print("\n[BAHASA] DISTRIBUSI BAHASA (setelah dedup original_id):")
    for lang, count in sorted(report["language_distribution_after_dedup_original_id"].items(), key=lambda x: -x[1]):
        print(f"   {lang:<10} {count:>5}")

    stats = report["body_length_stats"]
    print("\n[PANJANG] STATISTIK PANJANG TEKS (body):")
    print(f"   Min    : {stats['min']:,} karakter")
    print(f"   Maks   : {stats['max']:,} karakter")
    print(f"   Mean   : {stats['mean']:,.1f} karakter")
    print(f"   Median : {stats['median']:,.1f} karakter")
    print(f"   P25/P75: {stats['p25']:,.1f} / {stats['p75']:,.1f}")
    print(f"   Artikel < 200 karakter  : {stats['articles_lt_200_chars']}")
    print(f"   Artikel > 20.000 karakter: {stats['articles_gt_20000_chars']}")

    ent = report["entities_column"]
    print("\n[ENTITAS] KOLOM 'entities':")
    print(f"   Non-kosong : {ent['non_empty']}")
    print(f"   Kosong     : {ent['empty']}")
    print(f"   Catatan    : {ent['note']}")

    ref = report["reference_check"]
    print(f"\n[VERIFIKASI] {ref['status']}")
    if ref["deviations"]:
        for key, info in ref["deviations"].items():
            print(f"   [!] {key}: ekspektasi {info['expected']}, aktual {info['actual']} (selisih {info['diff']:+d})")

    print(f"\n{sep}\n")


# ──────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────

def main() -> None:
    logger.info("Memulai profiling dataset...")
    df = load_all_csvs()
    report = profile(df)

    # Simpan laporan JSON untuk reproducibility
    output_path = DATA_PROCESSED_DIR / "profiling_report.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    logger.info("Laporan profiling disimpan ke: %s", output_path)


if __name__ == "__main__":
    main()
