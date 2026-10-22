"""
pre_annotate.py — Pra-anotasi artikel dengan model baseline (model-assisted)
==========================================================================
Modul ini:
  1. Memilih subset representatif dari train, val, dan test split
  2. Menjalankan model NER baseline (spaCy) untuk menghasilkan kandidat anotasi
  3. Menerapkan penyesuaian aturan batas span sesuai annotation_guidelines.md
  4. Memvalidasi offset karakter agar pasti tepat: text[start:end] == ent_text
  5. Menyimpan hasil ke file JSONL dengan metadata provenance ("model-assisted")
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

import pandas as pd
import spacy

from src.config import (
    ANNOTATION_DIR,
    DATA_SPLITS_DIR,
    SEED,
    SUPPORTED_LABELS,
    TITLE_BODY_SEPARATOR,
)
from src.data_loader import load_all_csvs

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Gelar Malaysia yang harus disertakan ke dalam PERSON jika mendahului nama
MALAYSIAN_HONORIFICS = [
    "Datuk Seri",
    "Dato' Seri",
    "Datuk",
    "Dato'",
    "Tan Sri",
    "Tun",
    "Encik",
    "Puan",
    "Dr.",
    "Prof.",
]


def validate_spans(text: str, entities: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Validasi daftar entitas:
      - start < end
      - text[start:end] == ent['text']
      - Tidak ada overlapping spans (keep longest / earlier)
    """
    # Sort berdasarkan start lalu panjang span descending
    sorted_ents = sorted(entities, key=lambda e: (e["start"], -(e["end"] - e["start"])))
    valid_ents: list[dict[str, Any]] = []
    last_end = -1

    for ent in sorted_ents:
        start = ent["start"]
        end = ent["end"]
        ent_text = ent["text"]
        label = ent["label"]

        if start >= end:
            continue
        if start < last_end:
            # Overlap dengan span sebelumnya, lewati
            continue
        if text[start:end] != ent_text:
            continue
        if label not in SUPPORTED_LABELS:
            continue

        valid_ents.append(
            {
                "start": start,
                "end": end,
                "text": ent_text,
                "label": label,
            }
        )
        last_end = end

    return valid_ents


def adjust_spans_with_guidelines(text: str, entities: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Sesuaikan span kandidat agar mematuhi docs/annotation_guidelines.md:
      - Gelar kehormatan Malaysia digabung ke PERSON jika berada tepat di depan nama
      - "Berhad" / "Bhd" digabung ke ORG jika berada tepat setelah nama bank/organisasi
      - "Bank Negara Malaysia" dijadikan satu ORG utuh
    """
    adjusted = []
    for ent in entities:
        start = ent["start"]
        end = ent["end"]
        ent_text = ent["text"]
        label = ent["label"]

        # 1. Cek gelar di depan PERSON
        if label == "PERSON":
            for hon in sorted(MALAYSIAN_HONORIFICS, key=len, reverse=True):
                hon_len = len(hon)
                prefix_start = start - hon_len - 1
                if prefix_start >= 0 and text[prefix_start:start].strip() == hon:
                    # Ambil start baru yang mencakup gelar
                    new_start = prefix_start + text[prefix_start:start].find(hon)
                    start = new_start
                    ent_text = text[start:end]
                    break

        # 2. Cek "Berhad" / "Bhd" di belakang ORG
        if label == "ORG":
            suffix_candidate = text[end : end + 10]
            m = re.match(r"^\s+(Berhad|Bhd\.?|Group)", suffix_candidate, re.IGNORECASE)
            if m:
                end = end + m.end()
                ent_text = text[start:end]

        # 3. Handle kasus khusus "Bank Negara Malaysia"
        if "Bank Negara" in ent_text and "Malaysia" in text[end : end + 10]:
            idx = text[end : end + 10].find("Malaysia")
            end = end + idx + len("Malaysia")
            ent_text = text[start:end]
            label = "ORG"

        adjusted.append(
            {
                "start": start,
                "end": end,
                "text": ent_text,
                "label": label,
            }
        )

    return validate_spans(text, adjusted)


def pre_annotate_record(
    nlp: spacy.language.Language,
    record: dict[str, Any],
    max_text_chars: int = 3000,
) -> dict[str, Any]:
    """
    Lakukan pra-anotasi pada satu record artikel.
    """
    title = str(record.get("title", "")).strip()
    body = str(record.get("body", "")).strip()
    full_text = f"{title}{TITLE_BODY_SEPARATOR}{body}"

    # Potong jika teks terlalu panjang untuk peninjauan manusia yang reliabel
    if len(full_text) > max_text_chars:
        # Potong pada akhir kalimat terdekat di bawah max_text_chars
        truncated = full_text[:max_text_chars]
        last_period = truncated.rfind(".")
        if last_period > max_text_chars // 2:
            full_text = truncated[: last_period + 1]
        else:
            full_text = truncated

    doc = nlp(full_text)
    raw_entities = []
    for ent in doc.ents:
        if ent.label_ in SUPPORTED_LABELS:
            raw_entities.append(
                {
                    "start": ent.start_char,
                    "end": ent.end_char,
                    "text": ent.text,
                    "label": ent.label_,
                }
            )

    refined_entities = adjust_spans_with_guidelines(full_text, raw_entities)

    return {
        "original_id": record.get("original_id", ""),
        "source_name": record.get("source_name", ""),
        "source_file": record.get("source_file", ""),
        "language": record.get("language", "en"),
        "title": title,
        "text": full_text,
        "entities": refined_entities,
        "provenance": "model-assisted",
        "reviewed": False,
    }


def select_subset_for_split(
    df: pd.DataFrame,
    split_ids: list[str],
    n_samples: int,
    seed: int = SEED,
) -> pd.DataFrame:
    """
    Pilih subset artikel representatif dari daftar ID split tertentu.
    Memprioritaskan keragaman bahasa, panjang artikel yang wajar (200 - 3500 chars),
    dan keragaman sumber berita.
    """
    split_df = df[df["original_id"].isin(split_ids)].drop_duplicates(subset=["original_id"]).copy()

    # Hitung panjang body
    split_df["body_len"] = split_df["body"].astype(str).str.len()

    # Filter artikel yang terlalu pendek (< 150 char) kecuali untuk sampel ekstrem
    normal_df = split_df[(split_df["body_len"] >= 150) & (split_df["body_len"] <= 5000)]
    if len(normal_df) < n_samples:
        normal_df = split_df

    # Stratified sampling sederhana berdasarkan language jika memungkinkan
    sampled_list = []
    langs = normal_df["language"].unique()

    # Alokasikan kuota per bahasa
    for lang in langs:
        lang_df = normal_df[normal_df["language"] == lang]
        count = min(len(lang_df), max(2, int(n_samples * (len(lang_df) / len(normal_df)))))
        if count > 0:
            sampled_list.append(lang_df.sample(n=count, random_state=seed))

    combined = pd.concat(sampled_list).drop_duplicates(subset=["original_id"])

    # Jika kurang dari target n_samples, isi sisa dari sisa normal_df
    if len(combined) < n_samples:
        remaining = normal_df[~normal_df["original_id"].isin(combined["original_id"])]
        needed = min(len(remaining), n_samples - len(combined))
        if needed > 0:
            combined = pd.concat([combined, remaining.sample(n=needed, random_state=seed)])

    return combined.head(n_samples)


def run_pre_annotation(
    n_train: int = 70,
    n_val: int = 20,
    n_test: int = 20,
) -> dict[str, str]:
    """
    Jalankan proses pra-anotasi untuk train, val, dan test splits.
    Simpan hasil ke ANNOTATION_DIR dalam format JSONL.
    """
    ANNOTATION_DIR.mkdir(parents=True, exist_ok=True)

    splits_path = DATA_SPLITS_DIR / "splits.json"
    if not splits_path.exists():
        raise FileNotFoundError(f"{splits_path} tidak ditemukan. Jalankan dedup_split terlebih dahulu.")

    with open(splits_path, "r", encoding="utf-8") as f:
        splits = json.load(f)

    logger.info("Memuat data CSV...")
    df = load_all_csvs()

    logger.info("Memuat model spaCy en_core_web_sm...")
    nlp = spacy.load("en_core_web_sm")

    output_files = {}

    for split_name, n_target in [("train", n_train), ("val", n_val), ("test", n_test)]:
        split_ids = splits.get(split_name, [])
        logger.info("Memilih %d artikel representatif untuk split: %s...", n_target, split_name)
        subset_df = select_subset_for_split(df, split_ids, n_samples=n_target)

        out_path = ANNOTATION_DIR / f"{split_name}_preannotated.jsonl"
        records_written = 0
        total_ents = 0

        with open(out_path, "w", encoding="utf-8") as f:
            for _, row in subset_df.iterrows():
                annotated = pre_annotate_record(nlp, row.to_dict())
                f.write(json.dumps(annotated, ensure_ascii=False) + "\n")
                records_written += 1
                total_ents += len(annotated["entities"])

        logger.info(
            "Split %s selesai: %d artikel, %d total entitas -> %s",
            split_name,
            records_written,
            total_ents,
            out_path.name,
        )
        output_files[split_name] = str(out_path)

    return output_files


if __name__ == "__main__":
    run_pre_annotation()
