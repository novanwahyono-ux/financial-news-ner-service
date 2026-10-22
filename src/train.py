"""
train.py — Training / Fine-Tuning Pipeline NER (Fase 4)
======================================================
Modul ini:
  1. Memuat dataset terproses (data/processed/train.spacy dan val.spacy)
  2. Melakukan fine-tuning / adaptasi komponen NER spaCy dengan seed tetap
  3. Memonitor loss dan performa pada data validasi di setiap epoch
  4. Menyimpan checkpoint model terbaik ke models/ner_model/
  5. Mencatat metadata pelatihan ke models/training_meta.json
"""

from __future__ import annotations

import json
import logging
import random
import time
from pathlib import Path
from typing import Any

import spacy
from spacy.tokens import DocBin
from spacy.training import Example

from src.config import MODELS_DIR, SEED, DATA_PROCESSED_DIR

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def set_seed(seed: int = SEED) -> None:
    """Tetapkan random seed untuk reprodusibilitas penuh."""
    random.seed(seed)
    spacy.util.fix_random_seed(seed)


def load_docbin_examples(spacy_path: Path, nlp: spacy.language.Language) -> list[Example]:
    """Muat dokumen dari DocBin dan ubah menjadi list of spaCy Example."""
    if not spacy_path.exists():
        raise FileNotFoundError(f"File {spacy_path} tidak ditemukan.")
    
    doc_bin = DocBin().from_disk(spacy_path)
    docs = list(doc_bin.get_docs(nlp.vocab))
    examples = []
    for doc in docs:
        gold_dict = {
            "entities": [(ent.start_char, ent.end_char, ent.label_) for ent in doc.ents]
        }
        pred_doc = nlp.make_doc(doc.text)
        example = Example.from_dict(pred_doc, gold_dict)
        examples.append(example)
    return examples


def evaluate_examples(nlp: spacy.language.Language, examples: list[Example]) -> dict[str, Any]:
    """Evaluasi precision, recall, F1 pada sekumpulan Example."""
    scorer = nlp.evaluate(examples)
    ents_f = scorer.get("ents_f", 0.0)
    ents_p = scorer.get("ents_p", 0.0)
    ents_r = scorer.get("ents_r", 0.0)
    ents_per_type = scorer.get("ents_per_type", {})
    return {
        "f1": round(ents_f, 4),
        "precision": round(ents_p, 4),
        "recall": round(ents_r, 4),
        "per_type": ents_per_type,
    }


def train_ner_model(
    base_model_name: str = "en_core_web_sm",
    output_dir: Path = MODELS_DIR / "ner_model",
    n_epochs: int = 15,
    batch_size: int = 8,
    dropout: float = 0.2,
    seed: int = SEED,
) -> dict[str, Any]:
    """
    Fine-tuning komponen NER dari model dasar pada dataset berita perbankan.
    """
    set_seed(seed)
    start_time = time.time()
    logger.info("Memulai fine-tuning NER dengan seed=%d, epochs=%d...", seed, n_epochs)

    # 1. Muat model dasar
    try:
        nlp = spacy.load(base_model_name)
        logger.info("Model dasar '%s' berhasil dimuat.", base_model_name)
    except OSError:
        logger.info("Model dasar '%s' tidak ada di disk, menginisialisasi blank English pipeline...", base_model_name)
        nlp = spacy.blank("en")
        if "ner" not in nlp.pipe_names:
            nlp.add_pipe("ner", last=True)

    ner = nlp.get_pipe("ner")

    # 2. Muat data train dan val
    train_path = DATA_PROCESSED_DIR / "train.spacy"
    val_path = DATA_PROCESSED_DIR / "val.spacy"

    train_examples = load_docbin_examples(train_path, nlp)
    val_examples = load_docbin_examples(val_path, nlp)
    logger.info("Data dimuat: %d train examples, %d val examples.", len(train_examples), len(val_examples))

    # 3. Daftarkan label baru ke NER pipe
    for example in train_examples:
        for ent in example.reference.ents:
            ner.add_label(ent.label_)

    # 4. Freeze komponen lain selain NER untuk efisiensi & stabilitas
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
    logger.info("Freezing pipes selain ner: %s", other_pipes)

    optimizer = nlp.resume_training()

    history = []
    best_val_f1 = -1.0
    output_dir.mkdir(parents=True, exist_ok=True)

    with nlp.select_pipes(disable=other_pipes):
        for epoch in range(1, n_epochs + 1):
            random.shuffle(train_examples)
            losses: dict[str, float] = {}
            batches = spacy.util.minibatch(train_examples, size=batch_size)

            for batch in batches:
                nlp.update(batch, drop=dropout, losses=losses, sgd=optimizer)

            val_metrics = evaluate_examples(nlp, val_examples)
            ner_loss = round(losses.get("ner", 0.0), 4)

            epoch_log = {
                "epoch": epoch,
                "loss": ner_loss,
                "val_f1": val_metrics["f1"],
                "val_p": val_metrics["precision"],
                "val_r": val_metrics["recall"],
            }
            history.append(epoch_log)

            logger.info(
                "Epoch %02d/%02d | Loss: %.4f | Val F1: %.4f (P: %.4f, R: %.4f)",
                epoch,
                n_epochs,
                ner_loss,
                val_metrics["f1"],
                val_metrics["precision"],
                val_metrics["recall"],
            )

            # Simpan jika merupakan val F1 terbaik
            if val_metrics["f1"] >= best_val_f1:
                best_val_f1 = val_metrics["f1"]
                nlp.to_disk(output_dir)
                logger.info("--> Checkpoint terbaik disimpan (Val F1: %.4f)", best_val_f1)

    duration = round(time.time() - start_time, 2)
    logger.info("Training selesai dalam %.2f detik. Best Val F1: %.4f", duration, best_val_f1)

    meta_payload = {
        "model_type": "spacy_finetuned_ner",
        "base_model": base_model_name,
        "seed": seed,
        "n_epochs": n_epochs,
        "batch_size": batch_size,
        "dropout": dropout,
        "duration_seconds": duration,
        "train_examples_count": len(train_examples),
        "val_examples_count": len(val_examples),
        "best_val_f1": best_val_f1,
        "history": history,
        "artifact_path": str(output_dir),
    }

    meta_file = MODELS_DIR / "training_meta.json"
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(meta_payload, f, indent=2)

    logger.info("Metadata tersimpan di %s", meta_file)
    return meta_payload


if __name__ == "__main__":
    train_ner_model()
