"""
test_service.py — Tes Unit untuk ner_service.py (Fase 8)
========================================================
Memverifikasi:
  1. Presisi offset karakter: analyzed_text[start:end] == ent.text untuk semua entitas
  2. Determinisme: dua pemanggilan berturut-turut menghasilkan output yang persis identik
  3. Penanganan teks non-ASCII (Bahasa Melayu & Mandarin) tanpa error/crash
  4. Penanganan teks panjang (hingga puluhan ribu karakter) tanpa crash
  5. Konsistensi separator judul dan isi (\\n\\n)
  6. Filter label hanya menghasilkan SUPPORTED_LABELS
"""

from __future__ import annotations

import pytest

from src.config import SUPPORTED_LABELS, TITLE_BODY_SEPARATOR
from src.ner_service import build_analyzed_text, extract_entities


class TestNerService:
    def test_build_analyzed_text_separator(self):
        title = "Judul Berita"
        body = "Isi berita pertama."
        combined = build_analyzed_text(title, body)
        assert combined == f"Judul Berita{TITLE_BODY_SEPARATOR}Isi berita pertama."

    def test_exact_character_offsets(self):
        title = "Bank Negara Malaysia meeting in Kuala Lumpur"
        body = "Tan Sri Dato' Seri Zamzamzamin addressed shareholders on 15 September 2026."
        result = extract_entities(title, body)
        analyzed_text = build_analyzed_text(title, body)

        for ent in result["entities"]:
            start = ent["start"]
            end = ent["end"]
            ent_text = ent["text"]
            assert 0 <= start < end <= len(analyzed_text)
            assert analyzed_text[start:end] == ent_text, (
                f"Offset mismatch: analyzed_text[{start}:{end}]='{analyzed_text[start:end]}' != '{ent_text}'"
            )

    def test_determinism_identical_runs(self):
        title = "Maybank Group expands operations in Singapore and Indonesia"
        body = "Datuk Khairussaleh Ramli announced strategic investments for Q3 2026."

        res1 = extract_entities(title, body)
        res2 = extract_entities(title, body)

        assert res1["entities"] == res2["entities"]
        assert res1["meta"]["entity_count"] == res2["meta"]["entity_count"]

    def test_non_ascii_multilingual_handling(self):
        # Teks bahasa Melayu dengan tanda petik tunggal dan aksen serta karakter Mandarin
        title = "Dato' Seri Anwar bincang pelaburan bersama Maybank"
        body = "Perbincangan diadakan di Putrajaya pada 22 September 2026. 马来亚银行 (Maybank) catat prestasi baik."
        result = extract_entities(title, body)
        analyzed_text = build_analyzed_text(title, body)

        assert "entities" in result
        for ent in result["entities"]:
            start = ent["start"]
            end = ent["end"]
            assert analyzed_text[start:end] == ent["text"]

    def test_labels_within_supported_set(self):
        title = "Public Bank Berhad reports profit"
        body = "Founder Tan Sri Teh Hong Piow established Public Bank in Malaysia."
        result = extract_entities(title, body)

        for ent in result["entities"]:
            assert ent["label"] in SUPPORTED_LABELS

    def test_long_article_no_crash(self):
        # Artikel 15.000 karakter
        base_paragraph = "Maybank and CIMB Bank announced joint sustainable finance guidelines in Kuala Lumpur on 12 September 2026. "
        long_body = base_paragraph * 150
        result = extract_entities("Long Article Testing", long_body)

        assert "entities" in result
        assert result["meta"]["entity_count"] > 0
        assert result["meta"]["analyzed_length"] > 10000

    def test_empty_input_raises_value_error(self):
        with pytest.raises(ValueError, match="tidak boleh keduanya kosong"):
            extract_entities("", "   ")
