"""
converter.py — Konversi dataset JSONL / JSON ke spaCy DocBin (.spacy)
=====================================================================
Modul ini:
  1. Membaca file anotasi berformat JSONL atau evaluation_examples.json
  2. Memvalidasi tokenisasi dan batas span karakter (character offset -> token span)
  3. Membuang span yang beririsan atau tidak cocok dengan token boundary spaCy
  4. Menyimpan kumpulan Doc ke file biner spaCy DocBin (.spacy) untuk training
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import spacy
from spacy.tokens import DocBin

from src.config import ANNOTATION_DIR, DATA_PROCESSED_DIR, EVALUATION_FILE

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def records_to_docbin(
    records: list[dict[str, Any]],
    nlp: spacy.language.Language | None = None,
) -> tuple[DocBin, int, int]:
    """
    Konversi daftar record anotasi menjadi spaCy DocBin.

    Returns
    -------
    tuple of (doc_bin, total_valid_spans, total_skipped_spans)
    """
    if nlp is None:
        nlp = spacy.blank("en")

    doc_bin = DocBin()
    total_valid = 0
    total_skipped = 0

    for rec in records:
        text = rec.get("text", "")
        if not text:
            continue

        doc = nlp.make_doc(text)
        ents = []
        raw_ents = rec.get("entities", [])

        for e in raw_ents:
            start = e["start"]
            end = e["end"]
            label = e["label"]

            span = doc.char_span(start, end, label=label, alignment_mode="contract")
            if span is None:
                # Coba mode expand jika contract gagal karena tanda baca
                span = doc.char_span(start, end, label=label, alignment_mode="expand")

            if span is not None:
                ents.append(span)
                total_valid += 1
            else:
                total_skipped += 1
                logger.debug(
                    "Span dilewati karena tidak sejajar dengan token: '%s' [%d:%d]",
                    e.get("text", ""),
                    start,
                    end,
                )

        # Selesaikan overlap jika ada
        filtered_ents = spacy.util.filter_spans(ents)
        doc.ents = filtered_ents
        doc_bin.add(doc)

    return doc_bin, total_valid, total_skipped


def convert_all_annotations() -> dict[str, str]:
    """
    Konversi train, val, dan test/evaluation annotations ke file .spacy di data/processed/.
    """
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    nlp = spacy.blank("en")
    converted_paths = {}

    # 1. Konversi train
    train_jsonl = ANNOTATION_DIR / "train_preannotated.jsonl"
    if train_jsonl.exists():
        records = [json.loads(line) for line in train_jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
        doc_bin, valid, skipped = records_to_docbin(records, nlp)
        train_spacy_path = DATA_PROCESSED_DIR / "train.spacy"
        doc_bin.to_disk(train_spacy_path)
        logger.info("Train: %d docs, %d spans valid, %d spans skipped -> %s", len(records), valid, skipped, train_spacy_path.name)
        converted_paths["train"] = str(train_spacy_path)

    # 2. Konversi val
    val_jsonl = ANNOTATION_DIR / "val_preannotated.jsonl"
    if val_jsonl.exists():
        records = [json.loads(line) for line in val_jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
        doc_bin, valid, skipped = records_to_docbin(records, nlp)
        val_spacy_path = DATA_PROCESSED_DIR / "val.spacy"
        doc_bin.to_disk(val_spacy_path)
        logger.info("Val: %d docs, %d spans valid, %d spans skipped -> %s", len(records), valid, skipped, val_spacy_path.name)
        converted_paths["val"] = str(val_spacy_path)

    # 3. Konversi test / evaluation
    if EVALUATION_FILE.exists():
        with open(EVALUATION_FILE, "r", encoding="utf-8") as f:
            eval_data = json.load(f)
        records = eval_data.get("examples", [])
        doc_bin, valid, skipped = records_to_docbin(records, nlp)
        test_spacy_path = DATA_PROCESSED_DIR / "test.spacy"
        doc_bin.to_disk(test_spacy_path)
        logger.info("Test/Eval: %d docs, %d spans valid, %d spans skipped -> %s", len(records), valid, skipped, test_spacy_path.name)
        converted_paths["test"] = str(test_spacy_path)

    return converted_paths


if __name__ == "__main__":
    convert_all_annotations()
