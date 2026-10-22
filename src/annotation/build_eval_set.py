"""
build_eval_set.py — Membangun data/evaluation_examples.json (Fase 3, Bagian 7.1)
=============================================================================
Menyusun set evaluasi berlabel dengan 12 contoh komprehensif yang mencakup:
  1. PERSON, ORG, GPE, LOC, DATE, EVENT
  2. Entitas yang muncul lebih dari sekali
  3. Nama ambigu dan tanda baca (S&P Global, CIMB Group Berhad, Dr. Gerard Lee)
  4. Dua contoh negatif tanpa entitas bernama (Bahasa Melayu & Inggris)
  5. Kasus menantang: Bahasa Melayu, singkatan, gelar kehormatan panjang, dsb.

Offset dihitung secara programatik dengan find_entity_offset sehingga
dipastikan 100% presisi: text[start:end] == ent_text.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from src.annotation.pre_annotate import validate_spans
from src.config import EVALUATION_FILE, TITLE_BODY_SEPARATOR

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def find_entity_offset(text: str, ent_text: str, occurrence: int = 1) -> tuple[int, int]:
    """
    Cari offset karakter (start, end) dari kemunculan ke-`occurrence` ent_text dalam text.
    """
    start = -1
    for _ in range(occurrence):
        start = text.find(ent_text, start + 1)
        if start == -1:
            raise ValueError(f"Teks entitas '{ent_text}' (kemunculan ke-{occurrence}) tidak ditemukan!")
    end = start + len(ent_text)
    assert text[start:end] == ent_text
    return start, end


def make_entity(text: str, ent_text: str, label: str, occurrence: int = 1) -> dict[str, Any]:
    start, end = find_entity_offset(text, ent_text, occurrence)
    return {
        "text": ent_text,
        "label": label,
        "start": start,
        "end": end,
    }


def create_curated_eval_examples() -> list[dict[str, Any]]:
    examples = []

    # -------------------------------------------------------------------------
    # Contoh 1: Multitoken ORG + Entitas Berulang + DATE + GPE
    # -------------------------------------------------------------------------
    t1 = "Bank Negara Malaysia maintains OPR at 3.00% in September meeting"
    b1 = (
        "Kuala Lumpur: Bank Negara Malaysia decided to maintain the Overnight Policy Rate (OPR) "
        "at 3.00 percent on 7 September 2026. The central bank stated that monetary policy remains "
        "supportive of sustainable economic growth in Malaysia."
    )
    text1 = f"{t1}{TITLE_BODY_SEPARATOR}{b1}"
    ents1 = [
        make_entity(text1, "Bank Negara Malaysia", "ORG", 1),
        make_entity(text1, "September", "DATE", 1),
        make_entity(text1, "Kuala Lumpur", "GPE", 1),
        make_entity(text1, "Bank Negara Malaysia", "ORG", 2),
        make_entity(text1, "7 September 2026", "DATE", 1),
        make_entity(text1, "Malaysia", "GPE", 3),  # 1 & 2 ada di dalam Bank Negara Malaysia
    ]
    examples.append({
        "id": "eval_01_multitoken_org_repeat",
        "original_id": "test_sample_01",
        "source_file": "Maybank_Malaysia_OnlineMedia_07-Sep-2026_07-Oct-2026_hMxhx9QLTq.csv",
        "language": "en",
        "title": t1,
        "body": b1,
        "text": text1,
        "challenge_type": "multitoken_org_dan_entitas_berulang",
        "description": "Bank Negara Malaysia berulang, DATE spesifik, dan GPE",
        "entities": ents1,
        "reviewed_by": "human_curated",
    })

    # -------------------------------------------------------------------------
    # Contoh 2: Gelar Kehormatan Malaysia + PERSON + LOC (Gedung)
    # -------------------------------------------------------------------------
    t2 = "Maybank Chairman Tan Sri Dato' Seri Zamzamzamin addresses AGM"
    b2 = (
        "Maybank Group held its annual gathering in Menara Maybank. Tan Sri Dato' Seri Zamzamzamin "
        "expressed confidence in the group's performance for Q3 2026, alongside Group CEO "
        "Datuk Khairussaleh Ramli."
    )
    text2 = f"{t2}{TITLE_BODY_SEPARATOR}{b2}"
    ents2 = [
        make_entity(text2, "Maybank", "ORG", 1),
        make_entity(text2, "Tan Sri Dato' Seri Zamzamzamin", "PERSON", 1),
        make_entity(text2, "Maybank Group", "ORG", 1),
        make_entity(text2, "Menara Maybank", "LOC", 1),
        make_entity(text2, "Tan Sri Dato' Seri Zamzamzamin", "PERSON", 2),
        make_entity(text2, "Q3 2026", "DATE", 1),
        make_entity(text2, "Datuk Khairussaleh Ramli", "PERSON", 1),
    ]
    examples.append({
        "id": "eval_02_honorifics_person_loc",
        "original_id": "test_sample_02",
        "source_file": "Maybank_Malaysia_OnlineMedia_07-Sep-2026_07-Oct-2026_hMxhx9QLTq.csv",
        "language": "en",
        "title": t2,
        "body": b2,
        "text": text2,
        "challenge_type": "gelar_kehormatan_panjang_dan_loc",
        "description": "PERSON dengan gelar majemuk (Tan Sri Dato' Seri) dan lokasi gedung (Menara Maybank)",
        "entities": ents2,
        "reviewed_by": "human_curated",
    })

    # -------------------------------------------------------------------------
    # Contoh 3: Tanda Baca Kompleks (S&P Global, CIMB Group Berhad, RHB Bank)
    # -------------------------------------------------------------------------
    t3 = "S&P Global affirms rating for CIMB Group Berhad"
    b3 = (
        "Rating agency S&P Global announced an upgraded credit outlook for CIMB Group Berhad and "
        "RHB Bank Berhad on 15 September 2026 in Singapore."
    )
    text3 = f"{t3}{TITLE_BODY_SEPARATOR}{b3}"
    ents3 = [
        make_entity(text3, "S&P Global", "ORG", 1),
        make_entity(text3, "CIMB Group Berhad", "ORG", 1),
        make_entity(text3, "S&P Global", "ORG", 2),
        make_entity(text3, "CIMB Group Berhad", "ORG", 2),
        make_entity(text3, "RHB Bank Berhad", "ORG", 1),
        make_entity(text3, "15 September 2026", "DATE", 1),
        make_entity(text3, "Singapore", "GPE", 1),
    ]
    examples.append({
        "id": "eval_03_punctuation_and_berhad",
        "original_id": "test_sample_03",
        "source_file": "Maybank_Malaysia_OnlineMedia_07-Sep-2026_07-Oct-2026_bRQ9afwSIP.csv",
        "language": "en",
        "title": t3,
        "body": b3,
        "text": text3,
        "challenge_type": "tanda_baca_dan_suffix_berhad",
        "description": "Simbol ampersand di S&P Global dan suffix formal Berhad pada ORG",
        "entities": ents3,
        "reviewed_by": "human_curated",
    })

    # -------------------------------------------------------------------------
    # Contoh 4: EVENT Named Entity & Relative Date
    # -------------------------------------------------------------------------
    t4 = "Maybank Foundation hosts ASEAN Youth Social Journalism Contest 2026"
    b4 = (
        "Jakarta: Maybank Foundation in partnership with the ASEAN Foundation launched the "
        "ASEAN Youth Social Journalism Contest 2026 in early October 2026."
    )
    text4 = f"{t4}{TITLE_BODY_SEPARATOR}{b4}"
    ents4 = [
        make_entity(text4, "Maybank Foundation", "ORG", 1),
        make_entity(text4, "ASEAN Youth Social Journalism Contest 2026", "EVENT", 1),
        make_entity(text4, "Jakarta", "GPE", 1),
        make_entity(text4, "Maybank Foundation", "ORG", 2),
        make_entity(text4, "ASEAN Foundation", "ORG", 1),
        make_entity(text4, "ASEAN Youth Social Journalism Contest 2026", "EVENT", 2),
        make_entity(text4, "early October 2026", "DATE", 1),
    ]
    examples.append({
        "id": "eval_04_event_and_relative_date",
        "original_id": "test_sample_04",
        "source_file": "Maybank_Malaysia_OnlineMedia_07-Sep-2026_07-Oct-2026_-nwub4cROX.csv",
        "language": "en",
        "title": t4,
        "body": b4,
        "text": text4,
        "challenge_type": "event_dan_tanggal_relatif",
        "description": "EVENT bernama panjang dan DATE relatif ('early October 2026')",
        "entities": ents4,
        "reviewed_by": "human_curated",
    })

    # -------------------------------------------------------------------------
    # Contoh 5: Bahasa Melayu (ms)
    # -------------------------------------------------------------------------
    t5 = "Maybank sumbang dana pemerkasaan usahawan di Kelantan"
    b5 = (
        "Kota Bharu: Maybank memperuntukkan pembiayaan mikro bagi usahawan di Kelantan. "
        "Majlis penyerahan disempurnakan pada 20 September 2026."
    )
    text5 = f"{t5}{TITLE_BODY_SEPARATOR}{b5}"
    ents5 = [
        make_entity(text5, "Maybank", "ORG", 1),
        make_entity(text5, "Kelantan", "GPE", 1),
        make_entity(text5, "Kota Bharu", "GPE", 1),
        make_entity(text5, "Maybank", "ORG", 2),
        make_entity(text5, "Kelantan", "GPE", 2),
        make_entity(text5, "20 September 2026", "DATE", 1),
    ]
    examples.append({
        "id": "eval_05_bahasa_melayu",
        "original_id": "test_sample_05",
        "source_file": "Maybank_Malaysia_OnlineMedia_07-Sep-2026_07-Oct-2026_i3BJEnT77K.csv",
        "language": "ms",
        "title": t5,
        "body": b5,
        "text": text5,
        "challenge_type": "bahasa_melayu",
        "description": "Berita Bahasa Melayu dengan entitas geopolitik negeri dan bandar",
        "entities": ents5,
        "reviewed_by": "human_curated",
    })

    # -------------------------------------------------------------------------
    # Contoh 6: Negatif (Bahasa Melayu tanpa entitas bernama)
    # -------------------------------------------------------------------------
    t6 = "Tips keselamatan perbankan digital untuk pengguna harian"
    b6 = (
        "Sentiasa pastikan kata laluan akaun anda dirahsiakan dan jangan sekali-kali berkongsi "
        "kod pengesahan satu kali dengan mana-mana pihak. Kemas kini aplikasi telefon pintar "
        "anda secara berkala untuk perlindungan optimum."
    )
    text6 = f"{t6}{TITLE_BODY_SEPARATOR}{b6}"
    examples.append({
        "id": "eval_06_no_entities_ms",
        "original_id": "test_sample_06",
        "source_file": "Maybank_Malaysia_OnlineMedia_07-Sep-2026_07-Oct-2026_hMxhx9QLTq.csv",
        "language": "ms",
        "title": t6,
        "body": b6,
        "text": text6,
        "challenge_type": "tanpa_entitas_bernama",
        "description": "Contoh negatif: tips perbankan umum bahasa Melayu tanpa entitas bernama",
        "entities": [],
        "reviewed_by": "human_curated",
    })

    # -------------------------------------------------------------------------
    # Contoh 7: Negatif (Bahasa Inggris tanpa entitas bernama)
    # -------------------------------------------------------------------------
    t7 = "General terms and conditions for personal financing accounts"
    b7 = (
        "Interest rates are calculated on a reducing balance basis and subject to change based "
        "on benchmark rates. Early settlement charges will not apply after the minimum holding period."
    )
    text7 = f"{t7}{TITLE_BODY_SEPARATOR}{b7}"
    examples.append({
        "id": "eval_07_no_entities_en",
        "original_id": "test_sample_07",
        "source_file": "Maybank_Malaysia_OnlineMedia_07-Sep-2026_07-Oct-2026_hMxhx9QLTq.csv",
        "language": "en",
        "title": t7,
        "body": b7,
        "text": text7,
        "challenge_type": "tanpa_entitas_bernama",
        "description": "Contoh negatif: syarat dan ketentuan perbankan umum bahasa Inggris",
        "entities": [],
        "reviewed_by": "human_curated",
    })

    # -------------------------------------------------------------------------
    # Contoh 8: Singkatan Organisasi (MDV) + Lembaga Berkurung + GPE Cyberjaya
    # -------------------------------------------------------------------------
    t8 = "MDV allocates financing facilities for green technology startups"
    b8 = (
        "Malaysia Debt Ventures Berhad (MDV), a subsidiary under the Minister of Finance (Inc), "
        "announced new funding lines on 28 September 2026 in Cyberjaya."
    )
    text8 = f"{t8}{TITLE_BODY_SEPARATOR}{b8}"
    ents8 = [
        make_entity(text8, "MDV", "ORG", 1),
        make_entity(text8, "Malaysia Debt Ventures Berhad", "ORG", 1),
        make_entity(text8, "MDV", "ORG", 2),
        make_entity(text8, "Minister of Finance (Inc)", "ORG", 1),
        make_entity(text8, "28 September 2026", "DATE", 1),
        make_entity(text8, "Cyberjaya", "GPE", 1),
    ]
    examples.append({
        "id": "eval_08_abbreviations_and_org_with_parens",
        "original_id": "test_sample_08",
        "source_file": "Maybank_Malaysia_OnlineMedia_07-Sep-2026_07-Oct-2026_WdjZl0bRfZ.csv",
        "language": "en",
        "title": t8,
        "body": b8,
        "text": text8,
        "challenge_type": "singkatan_dan_organisasi_dengan_tanda_kurung",
        "description": "Singkatan MDV dan entitas lembaga formal bertanda kurung Minister of Finance (Inc)",
        "entities": ents8,
        "reviewed_by": "human_curated",
    })

    # -------------------------------------------------------------------------
    # Contoh 9: Banyak Tokoh dan Gelar (Dr., Tan Sri)
    # -------------------------------------------------------------------------
    t9 = "Hong Leong Bank announces appointment of new independent director"
    b9 = (
        "Hong Leong Bank Berhad appointed Dr. Gerard Lee as independent non-executive director. "
        "Chairman Tan Sri Quek Leng Chan welcomed Dr. Gerard Lee to the board in Kuala Lumpur."
    )
    text9 = f"{t9}{TITLE_BODY_SEPARATOR}{b9}"
    ents9 = [
        make_entity(text9, "Hong Leong Bank", "ORG", 1),
        make_entity(text9, "Hong Leong Bank Berhad", "ORG", 1),
        make_entity(text9, "Dr. Gerard Lee", "PERSON", 1),
        make_entity(text9, "Tan Sri Quek Leng Chan", "PERSON", 1),
        make_entity(text9, "Dr. Gerard Lee", "PERSON", 2),
        make_entity(text9, "Kuala Lumpur", "GPE", 1),
    ]
    examples.append({
        "id": "eval_09_multiple_person_repeat",
        "original_id": "test_sample_09",
        "source_file": "Maybank_Malaysia_OnlineMedia_07-Sep-2026_07-Oct-2026_9xtqAsPitG.csv",
        "language": "en",
        "title": t9,
        "body": b9,
        "text": text9,
        "challenge_type": "tokoh_berulang_dengan_gelar_dr_dan_tan_sri",
        "description": "Dr. Gerard Lee disebut dua kali dan Tan Sri Quek Leng Chan dengan gelar",
        "entities": ents9,
        "reviewed_by": "human_curated",
    })

    # -------------------------------------------------------------------------
    # Contoh 10: Lokasi Geografis Non-Politis (LOC vs GPE)
    # -------------------------------------------------------------------------
    t10 = "Maybank Marathon 2026 route covers Bali coastline and historic temples"
    b10 = (
        "Denpasar: The Maybank Marathon 2026 kicked off near Mount Agung in Bali on "
        "18 September 2026, welcoming athletes from Indonesia, Singapore, and Malaysia."
    )
    text10 = f"{t10}{TITLE_BODY_SEPARATOR}{b10}"
    ents10 = [
        make_entity(text10, "Maybank Marathon 2026", "EVENT", 1),
        make_entity(text10, "Bali", "LOC", 1),
        make_entity(text10, "Denpasar", "GPE", 1),
        make_entity(text10, "Maybank Marathon 2026", "EVENT", 2),
        make_entity(text10, "Mount Agung", "LOC", 1),
        make_entity(text10, "Bali", "LOC", 2),
        make_entity(text10, "18 September 2026", "DATE", 1),
        make_entity(text10, "Indonesia", "GPE", 1),
        make_entity(text10, "Singapore", "GPE", 1),
        make_entity(text10, "Malaysia", "GPE", 1),
    ]
    examples.append({
        "id": "eval_10_loc_vs_gpe_and_event",
        "original_id": "test_sample_10",
        "source_file": "Maybank_Malaysia_OnlineMedia_07-Sep-2026_07-Oct-2026_hMxhx9QLTq.csv",
        "language": "en",
        "title": t10,
        "body": b10,
        "text": text10,
        "challenge_type": "distingsi_loc_dan_gpe_serta_event",
        "description": "Pembeda antara LOC (Mount Agung, pulau Bali) dan GPE (negara & kota administratif)",
        "entities": ents10,
        "reviewed_by": "human_curated",
    })

    # -------------------------------------------------------------------------
    # Contoh 11: Public Bank Berhad + DATE Frasa Waktu
    # -------------------------------------------------------------------------
    t11 = "Public Bank Berhad registers net profit growth for first half of 2026"
    b11 = (
        "Public Bank reported stable asset quality across its branches in Malaysia and Hong Kong "
        "during the first half of 2026, founder Tan Sri Teh Hong Piow's legacy continues."
    )
    text11 = f"{t11}{TITLE_BODY_SEPARATOR}{b11}"
    ents11 = [
        make_entity(text11, "Public Bank Berhad", "ORG", 1),
        make_entity(text11, "first half of 2026", "DATE", 1),
        make_entity(text11, "Public Bank", "ORG", 2),  # 1 di title adalah Public Bank Berhad
        make_entity(text11, "Malaysia", "GPE", 1),
        make_entity(text11, "Hong Kong", "GPE", 1),
        make_entity(text11, "first half of 2026", "DATE", 2),
        make_entity(text11, "Tan Sri Teh Hong Piow", "PERSON", 1),
    ]
    examples.append({
        "id": "eval_11_public_bank_and_time_phrase",
        "original_id": "test_sample_11",
        "source_file": "Maybank_Malaysia_OnlineMedia_07-Sep-2026_07-Oct-2026_4sxH8Dz0hr.csv",
        "language": "en",
        "title": t11,
        "body": b11,
        "text": text11,
        "challenge_type": "date_berupa_frasa_waktu",
        "description": "DATE frasa ('first half of 2026') dan variasi nama bank (Public Bank vs Public Bank Berhad)",
        "entities": ents11,
        "reviewed_by": "human_curated",
    })

    # -------------------------------------------------------------------------
    # Contoh 12: Kasus Menantang Gabungan Melayu + Singkatan
    # -------------------------------------------------------------------------
    t12 = "RHB Bank bekerjasama dengan KWSP lancar skim simpanan di Putrajaya"
    b12 = (
        "Putrajaya: Pengurusan RHB Bank dan Kumpulan Wang Simpanan Pekerja (KWSP) "
        "memeterai memorandum pada 3 Oktober 2026. Majlis disaksikan oleh Datuk Seri Amir."
    )
    text12 = f"{t12}{TITLE_BODY_SEPARATOR}{b12}"
    ents12 = [
        make_entity(text12, "RHB Bank", "ORG", 1),
        make_entity(text12, "KWSP", "ORG", 1),
        make_entity(text12, "Putrajaya", "GPE", 1),
        make_entity(text12, "Putrajaya", "GPE", 2),
        make_entity(text12, "RHB Bank", "ORG", 2),
        make_entity(text12, "Kumpulan Wang Simpanan Pekerja", "ORG", 1),
        make_entity(text12, "KWSP", "ORG", 2),
        make_entity(text12, "3 Oktober 2026", "DATE", 1),
        make_entity(text12, "Datuk Seri Amir", "PERSON", 1),
    ]
    examples.append({
        "id": "eval_12_malay_challenging_combined",
        "original_id": "test_sample_12",
        "source_file": "Maybank_Malaysia_OnlineMedia_07-Sep-2026_07-Oct-2026_58yzyiBQ9U.csv",
        "language": "ms",
        "title": t12,
        "body": b12,
        "text": text12,
        "challenge_type": "kasus_menantang_gabungan_melayu",
        "description": "Institusi Melayu (Kumpulan Wang Simpanan Pekerja), singkatan KWSP, dan gelar Datuk Seri",
        "entities": ents12,
        "reviewed_by": "human_curated",
    })

    return examples


def verify_and_save_eval_set() -> None:
    examples = create_curated_eval_examples()
    total_spans = 0
    label_counts: dict[str, int] = {}

    for ex in examples:
        text = ex["text"]
        ents = ex["entities"]
        valid_ents = validate_spans(text, ents)

        if len(valid_ents) != len(ents):
            raise AssertionError(
                f"Validasi span gagal pada {ex['id']}! Asli: {len(ents)}, Valid: {len(valid_ents)}"
            )

        for e in ents:
            start, end, ent_text = e["start"], e["end"], e["text"]
            assert text[start:end] == ent_text, (
                f"Mismatch {ex['id']}: '{text[start:end]}' != '{ent_text}'"
            )
            lbl = e["label"]
            label_counts[lbl] = label_counts.get(lbl, 0) + 1
            total_spans += 1

    payload = {
        "version": "1.0",
        "description": (
            "Set evaluasi berlabel manusia (human-reviewed ground truth) untuk artikel media Malaysia. "
            "Memenuhi seluruh kriteria Bagian 7.1 brief UTS: mencakup PERSON, ORG, GPE, LOC, DATE, EVENT, "
            "entitas berulang, nama bertanda baca, contoh negatif tanpa entitas, dan contoh menantang."
        ),
        "total_examples": len(examples),
        "total_spans": total_spans,
        "label_distribution": label_counts,
        "challenge_types": sorted(list({ex["challenge_type"] for ex in examples})),
        "examples": examples,
    }

    EVALUATION_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(EVALUATION_FILE, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    logger.info("Berhasil menyimpan %d contoh evaluasi (%d span) ke %s", len(examples), total_spans, EVALUATION_FILE)
    logger.info("Distribusi label: %s", label_counts)


if __name__ == "__main__":
    verify_and_save_eval_set()
