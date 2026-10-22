"""
ner_service.py — Modul NER: muat model dan ekstrak entitas
===========================================================
Modul ini:
  1. Memuat model spaCy yang dikonfigurasi (sekali saat startup)
  2. Menggabungkan title + body dengan separator terdokumentasi
  3. Menjalankan pipeline NER dan mengembalikan entitas dengan offset karakter
  4. Memastikan output deterministik (no sampling)

Penggunaan:
    from src.ner_service import extract_entities
    result = extract_entities(title="...", body="...")
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
    """
    Muat model spaCy sekali dan cache-nya.
    Gunakan `disable` untuk komponen yang tidak dibutuhkan (lebih cepat).
    """
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
# Fungsi publik
# ──────────────────────────────────────────────────────────────

def build_analyzed_text(title: str, body: str) -> str:
    """
    Gabungkan title dan body dengan separator terdokumentasi.
    Format: title + "\\n\\n" + body
    Offset entitas dihitung pada teks gabungan ini.
    """
    return title + TITLE_BODY_SEPARATOR + body


def extract_entities(title: str, body: str) -> dict[str, Any]:
    """
    Jalankan NER pada teks gabungan title + body.

    Parameters
    ----------
    title : str
        Judul artikel.
    body : str
        Isi artikel.

    Returns
    -------
    dict dengan:
        - entities: list of {text, label, start, end}
        - meta: {entity_count, model, analyzed_length, title_body_separator}

    Raises
    ------
    ValueError
        Jika kedua title dan body kosong, atau melebihi MAX_INPUT_CHARS.
    """
    # Validasi (juga dilakukan di API layer, tapi service harus mandiri)
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

    entities = []
    for ent in doc.ents:
        # Filter hanya label yang didukung
        if ent.label_ not in SUPPORTED_LABELS:
            continue
        entities.append(
            {
                "text": ent.text,
                "label": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char,
            }
        )

    return {
        "entities": entities,
        "meta": {
            "entity_count": len(entities),
            "model": MODEL_NAME,
            "analyzed_length": len(analyzed_text),
            "title_body_separator": repr(TITLE_BODY_SEPARATOR),
        },
    }
