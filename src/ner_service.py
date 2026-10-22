"""
ner_service.py — Modul NER: muat model dan ekstrak entitas
===========================================================
Modul ini:
  1. Memuat model spaCy yang dikonfigurasi (sekali saat startup)
  2. Menggabungkan title + body dengan separator terdokumentasi
  3. Menjalankan pipeline NER dan mengembalikan entitas dengan offset karakter
  4. Melakukan post-processing untuk menggabungkan entitas PERSON yang terpisah
  5. Memastikan output deterministik (no sampling)
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

import spacy
from spacy.language import Language

from src.config import (
    MAX_INPUT_CHARS,
    MODEL_NAME,
    SUPPORTED_LABELS,
    TITLE_BODY_SEPARATOR,
)

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# Model loading (lazy singleton)
# ──────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _load_model() -> Language:
    logger.info("Memuat model NER: %s", MODEL_NAME)
    try:
        nlp = spacy.load(MODEL_NAME, disable=["tagger", "parser", "lemmatizer"])
    except OSError as exc:
        raise RuntimeError(
            f"Model '{MODEL_NAME}' tidak ditemukan. "
            f"Jalankan: python -m spacy download {MODEL_NAME}\n{exc}"
        ) from exc
    logger.info("Model '%s' berhasil dimuat.", MODEL_NAME)
    return nlp


# ──────────────────────────────────────────────────────────────
# Helper: Merge adjacent PERSON entities
# ──────────────────────────────────────────────────────────────

def _merge_adjacent_person_entities(entities: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Menggabungkan entitas PERSON yang terpisah berurutan akibat
    keterbatasan batas token/gelar (misal: "Tan Sri Dato'" + "Seri Zamzamzamin").
    """
    if not entities:
        return []

    merged = []
    i = 0
    while i < len(entities):
        curr = entities[i].copy()
        while i + 1 < len(entities):
            nxt = entities[i + 1]
            # Jika entitas saat ini & berikutnya sama-sama PERSON dan posisinya berdampingan (spasi <= 1)
            if (
                curr["label"] == "PERSON"
                and nxt["label"] == "PERSON"
                and (nxt["start"] - curr["end"]) <= 1
            ):
                gap = " " if (nxt["start"] - curr["end"]) == 1 else ""
                curr["text"] = curr["text"] + gap + nxt["text"]
                curr["end"] = nxt["end"]
                i += 1
            else:
                break
        merged.append(curr)
        i += 1
    return merged


# ──────────────────────────────────────────────────────────────
# Fungsi publik
# ──────────────────────────────────────────────────────────────

def build_analyzed_text(title: str, body: str) -> str:
    return title + TITLE_BODY_SEPARATOR + body


def extract_entities(title: str, body: str) -> dict[str, Any]:
    title = title or ""
    body = body or ""

    if not title.strip() and not body.strip():
        raise ValueError("title dan body tidak boleh keduanya kosong.")

    analyzed_text = build_analyzed_text(title, body)

    if len(analyzed_text) > MAX_INPUT_CHARS:
        raise ValueError(
            f"Input melebihi batas maksimum {MAX_INPUT_CHARS:,} karakter "
            f"(aktual: {len(analyzed_text):,})."
        )

    nlp = _load_model()
    doc = nlp(analyzed_text)

    raw_entities = []
    for ent in doc.ents:
        if ent.label_ not in SUPPORTED_LABELS:
            continue
        raw_entities.append(
            {
                "text": ent.text,
                "label": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char,
            }
        )

    # Gabungkan entitas PERSON yang terpisah
    final_entities = _merge_adjacent_person_entities(raw_entities)

    return {
        "entities": final_entities,
        "meta": {
            "entity_count": len(final_entities),
            "model": MODEL_NAME,
            "analyzed_length": len(analyzed_text),
            "title_body_separator": repr(TITLE_BODY_SEPARATOR),
        },
    }