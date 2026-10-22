"""
api.py — FastAPI Backend Service untuk Article NER (Fase 6 & 7)
==============================================================
Menyediakan:
  - GET  /                  : Panel Web interaktif (HTML/CSS/JS modern)
  - POST /extract-entities  : Endpoint ekstraksi entitas NER (JSON)
  - GET  /health            : Endpoint health check sistem

Menjamin kepatuhan penuh terhadap ketentuan brief UTS:
  - Input JSON: {"title": str, "body": str}
  - Status HTTP 400 untuk input invalid (bukan string, keduanya kosong, malformed, overlimit)
  - Output JSON: {"entities": [...], "meta": {...}}
  - Determinisme dan validasi offset karakter
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from src.config import MAX_INPUT_CHARS, MODEL_NAME, TITLE_BODY_SEPARATOR
from src.ner_service import extract_entities

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Article Named-Entity Recognition Service",
    description="Layanan ekstraksi named entity dari judul dan isi artikel berita online Malaysia.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────────────────────
# Custom Exception Handler agar selalu HTTP 400 untuk bad request
# ──────────────────────────────────────────────────────────────
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={
            "error": "Bad Request",
            "message": "Validasi input gagal. Pastikan format JSON benar, field 'title' dan 'body' bertipe string.",
            "details": exc.errors(),
        },
    )


@app.exception_handler(json.JSONDecodeError)
async def json_decode_exception_handler(request: Request, exc: json.JSONDecodeError) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={
            "error": "Bad Request",
            "message": f"Format JSON tidak valid (malformed): {exc.msg}",
        },
    )


# ──────────────────────────────────────────────────────────────
# Model Request & Endpoint
# ──────────────────────────────────────────────────────────────
class ArticleRequest(BaseModel):
    title: Any = Field(..., description="Judul artikel (wajib string)")
    body: Any = Field(..., description="Isi artikel (wajib string)")


@app.get("/health")
def health_check() -> dict[str, str]:
    return {
        "status": "ok",
        "service": "Article NER Service",
        "model": MODEL_NAME,
        "max_input_chars": str(MAX_INPUT_CHARS),
    }


@app.post("/extract-entities")
async def extract_entities_endpoint(request: Request) -> JSONResponse:
    # 1. Parse raw body untuk mendeteksi malformed JSON
    try:
        raw_body = await request.body()
        if not raw_body:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"error": "Bad Request", "message": "Body request tidak boleh kosong."},
            )
        data = json.loads(raw_body)
    except Exception as exc:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"error": "Bad Request", "message": f"JSON malformed atau tidak dapat di-parse: {str(exc)}"},
        )

    if not isinstance(data, dict):
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"error": "Bad Request", "message": "Payload JSON harus berupa object (dictionary)."},
        )

    # 2. Validasi keberadaan dan tipe field
    if "title" not in data or "body" not in data:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"error": "Bad Request", "message": "Payload harus memuat field 'title' dan 'body'."},
        )

    title = data.get("title")
    body = data.get("body")

    if not isinstance(title, str) or not isinstance(body, str):
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"error": "Bad Request", "message": "Field 'title' dan 'body' keduanya harus bertipe string."},
        )

    # 3. Validasi tidak boleh keduanya kosong/whitespace
    if not title.strip() and not body.strip():
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"error": "Bad Request", "message": "Field 'title' dan 'body' tidak boleh keduanya kosong."},
        )

    # 4. Validasi batas panjang maksimum input
    analyzed_text = f"{title}{TITLE_BODY_SEPARATOR}{body}"
    if len(analyzed_text) > MAX_INPUT_CHARS:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "error": "Bad Request",
                "message": (
                    f"Panjang gabungan teks ({len(analyzed_text):,} karakter) melebihi batas "
                    f"maksimum yang diizinkan ({MAX_INPUT_CHARS:,} karakter)."
                ),
            },
        )

    # 5. Ekstraksi entitas dengan model NER
    try:
        result = extract_entities(title=title, body=body)
        return JSONResponse(status_code=status.HTTP_200_OK, content=result)
    except Exception as exc:
        logger.error("Error pada inferensi NER: %s", exc, exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error": "Internal Server Error", "message": str(exc)},
        )


# ──────────────────────────────────────────────────────────────
# Panel Web UI (HTML + CSS Modern + JavaScript)
# ──────────────────────────────────────────────────────────────
INDEX_HTML = """<!DOCTYPE html>
<html lang="id">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Article NER Service — Pemantauan Berita Online Malaysia</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Fira+Code:wght@400;500&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg: #0b0f19;
      --card-bg: #111827;
      --card-border: #1f2937;
      --text: #f3f4f6;
      --text-muted: #9ca3af;
      --primary: #3b82f6;
      --primary-hover: #2563eb;
      --accent: #10b981;
      --danger: #ef4444;
      --org-color: #3b82f6;
      --person-color: #10b981;
      --gpe-color: #f59e0b;
      --loc-color: #8b5cf6;
      --date-color: #ec4899;
      --event-color: #06b6d4;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.6;
      padding: 24px;
    }
    .container {
      max-width: 1200px;
      margin: 0 auto;
    }
    header {
      margin-bottom: 24px;
      padding-bottom: 16px;
      border-bottom: 1px solid var(--card-border);
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 16px;
    }
    h1 {
      font-size: 1.6rem;
      font-weight: 700;
      color: #fff;
      display: flex;
      align-items: center;
      gap: 10px;
    }
    .badge {
      display: inline-block;
      padding: 4px 10px;
      border-radius: 9999px;
      font-size: 0.75rem;
      font-weight: 600;
      background: #1e3a8a;
      color: #93c5fd;
    }
    .grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 24px;
    }
    @media (max-width: 900px) {
      .grid { grid-template-columns: 1fr; }
    }
    .card {
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 12px;
      padding: 20px;
      box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
    }
    .card-title {
      font-size: 1.1rem;
      font-weight: 600;
      margin-bottom: 16px;
      color: #e5e7eb;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    .form-group {
      margin-bottom: 16px;
    }
    label {
      display: block;
      font-size: 0.85rem;
      font-weight: 500;
      color: var(--text-muted);
      margin-bottom: 6px;
    }
    input[type="text"], textarea {
      width: 100%;
      background: #0f172a;
      border: 1px solid var(--card-border);
      border-radius: 8px;
      padding: 10px 14px;
      color: #fff;
      font-family: inherit;
      font-size: 0.9rem;
      transition: border-color 0.15s ease;
    }
    input[type="text"]:focus, textarea:focus {
      outline: none;
      border-color: var(--primary);
    }
    textarea {
      min-height: 220px;
      resize: vertical;
    }
    .samples {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      margin-bottom: 16px;
    }
    .sample-btn {
      background: #1f2937;
      border: 1px solid #374151;
      color: #d1d5db;
      font-size: 0.75rem;
      padding: 4px 10px;
      border-radius: 6px;
      cursor: pointer;
      transition: all 0.15s ease;
    }
    .sample-btn:hover {
      background: #374151;
      color: #fff;
    }
    .btn-submit {
      background: var(--primary);
      color: #fff;
      border: none;
      border-radius: 8px;
      padding: 12px 20px;
      font-size: 0.95rem;
      font-weight: 600;
      cursor: pointer;
      width: 100%;
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 8px;
      transition: background 0.15s ease;
    }
    .btn-submit:hover:not(:disabled) {
      background: var(--primary-hover);
    }
    .btn-submit:disabled {
      opacity: 0.6;
      cursor: not-allowed;
    }
    .spinner {
      width: 18px;
      height: 18px;
      border: 2px solid rgba(255,255,255,0.3);
      border-radius: 50%;
      border-top-color: #fff;
      animation: spin 0.8s linear infinite;
      display: none;
    }
    @keyframes spin {
      to { transform: rotate(360deg); }
    }
    .alert {
      padding: 12px 16px;
      border-radius: 8px;
      margin-bottom: 16px;
      font-size: 0.85rem;
      display: none;
    }
    .alert-danger {
      background: rgba(239, 68, 68, 0.15);
      border: 1px solid rgba(239, 68, 68, 0.4);
      color: #fca5a5;
    }
    .stats-bar {
      display: flex;
      gap: 12px;
      margin-bottom: 16px;
      flex-wrap: wrap;
    }
    .stat-pill {
      background: #1f2937;
      padding: 6px 12px;
      border-radius: 6px;
      font-size: 0.8rem;
    }
    .stat-pill strong { color: #fff; }
    .table-container {
      overflow-x: auto;
      max-height: 280px;
      border: 1px solid var(--card-border);
      border-radius: 8px;
      margin-bottom: 20px;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 0.85rem;
      text-align: left;
    }
    th {
      background: #1f2937;
      color: #d1d5db;
      font-weight: 600;
      padding: 10px 14px;
      position: sticky;
      top: 0;
    }
    td {
      padding: 8px 14px;
      border-top: 1px solid var(--card-border);
      color: #e5e7eb;
    }
    tr:hover td {
      background: rgba(255,255,255,0.02);
    }
    .tag {
      display: inline-block;
      padding: 2px 8px;
      border-radius: 4px;
      font-size: 0.72rem;
      font-weight: 600;
    }
    .tag-ORG { background: rgba(59, 130, 246, 0.2); color: #93c5fd; border: 1px solid rgba(59, 130, 246, 0.4); }
    .tag-PERSON { background: rgba(16, 185, 129, 0.2); color: #6ee7b7; border: 1px solid rgba(16, 185, 129, 0.4); }
    .tag-GPE { background: rgba(245, 158, 11, 0.2); color: #fde68a; border: 1px solid rgba(245, 158, 11, 0.4); }
    .tag-LOC { background: rgba(139, 92, 246, 0.2); color: #c4b5fd; border: 1px solid rgba(139, 92, 246, 0.4); }
    .tag-DATE { background: rgba(236, 72, 153, 0.2); color: #fbcfe8; border: 1px solid rgba(236, 72, 153, 0.4); }
    .tag-EVENT { background: rgba(6, 182, 212, 0.2); color: #a5f3fc; border: 1px solid rgba(6, 182, 212, 0.4); }
    .highlight-box {
      background: #0f172a;
      border: 1px solid var(--card-border);
      border-radius: 8px;
      padding: 16px;
      font-size: 0.9rem;
      line-height: 1.8;
      max-height: 320px;
      overflow-y: auto;
      white-space: pre-wrap;
      word-break: break-word;
    }
    .hl-span {
      padding: 2px 4px;
      border-radius: 4px;
      font-weight: 500;
      position: relative;
    }
    .hl-span::after {
      content: attr(data-label);
      font-size: 0.65rem;
      margin-left: 4px;
      font-weight: 700;
      opacity: 0.85;
      text-transform: uppercase;
    }
    .hl-ORG { background: rgba(59, 130, 246, 0.3); border-bottom: 2px solid var(--org-color); }
    .hl-PERSON { background: rgba(16, 185, 129, 0.3); border-bottom: 2px solid var(--person-color); }
    .hl-GPE { background: rgba(245, 158, 11, 0.3); border-bottom: 2px solid var(--gpe-color); }
    .hl-LOC { background: rgba(139, 92, 246, 0.3); border-bottom: 2px solid var(--loc-color); }
    .hl-DATE { background: rgba(236, 72, 153, 0.3); border-bottom: 2px solid var(--date-color); }
    .hl-EVENT { background: rgba(6, 182, 212, 0.3); border-bottom: 2px solid var(--event-color); }
    .empty-state {
      text-align: center;
      padding: 40px 20px;
      color: var(--text-muted);
      font-size: 0.9rem;
    }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <div>
        <h1>📰 Article NER Service</h1>
        <p style="color: var(--text-muted); font-size: 0.85rem; margin-top: 4px;">
          Sistem Ekstraksi Named Entity Pengawasan Berita Perbankan Malaysia
        </p>
      </div>
      <div>
        <span class="badge" id="modelBadge">Model: models/ner_model</span>
      </div>
    </header>

    <div class="grid">
      <!-- Panel Input -->
      <div class="card">
        <div class="card-title">
          <span>Input Artikel</span>
        </div>

        <div class="samples">
          <span style="font-size: 0.75rem; color: var(--text-muted); align-self: center;">Contoh cepat:</span>
          <button class="sample-btn" onclick="loadSample(1)">Bank Negara Malaysia</button>
          <button class="sample-btn" onclick="loadSample(2)">Gelar & Tokoh Bank</button>
          <button class="sample-btn" onclick="loadSample(3)">S&P & CIMB Berhad</button>
          <button class="sample-btn" onclick="loadSample(4)">Bahasa Melayu</button>
          <button class="sample-btn" onclick="loadSample(5)">Contoh Negatif</button>
        </div>

        <div id="errorAlert" class="alert alert-danger"></div>

        <form id="nerForm" onsubmit="handleExtract(event)">
          <div class="form-group">
            <label for="titleInput">Judul Artikel (Title):</label>
            <input type="text" id="titleInput" placeholder="Masukkan judul artikel berita..." required>
          </div>

          <div class="form-group">
            <label for="bodyInput">Isi Artikel (Body):</label>
            <textarea id="bodyInput" placeholder="Masukkan isi lengkap artikel berita..." required></textarea>
          </div>

          <button type="submit" id="submitBtn" class="btn-submit">
            <div id="btnSpinner" class="spinner"></div>
            <span id="btnText">Ekstrak Named Entity</span>
          </button>
        </form>
      </div>

      <!-- Panel Hasil -->
      <div class="card">
        <div class="card-title">
          <span>Hasil Deteksi Entitas</span>
          <span id="resultMeta" style="font-size: 0.8rem; font-weight: normal; color: var(--text-muted);"></span>
        </div>

        <div id="statsBar" class="stats-bar" style="display: none;">
          <div class="stat-pill">Total Entitas: <strong id="totalEntCount">0</strong></div>
          <div class="stat-pill">Panjang Teks: <strong id="textLengthCount">0</strong> char</div>
        </div>

        <div id="tableSection" style="display: none;">
          <div class="table-container">
            <table>
              <thead>
                <tr>
                  <th>Teks Entitas</th>
                  <th>Kategori</th>
                  <th>Start</th>
                  <th>End</th>
                </tr>
              </thead>
              <tbody id="entitiesTableBody"></tbody>
            </table>
          </div>

          <label style="margin-bottom: 8px;">Penyorotan Teks (Visual Highlight):</label>
          <div id="highlightBox" class="highlight-box"></div>
        </div>

        <div id="emptyState" class="empty-state">
          Masukkan judul dan isi artikel di sebelah kiri, lalu tekan <strong>Ekstrak Named Entity</strong>.
        </div>
      </div>
    </div>
  </div>

  <script>
    const SAMPLES = {
      1: {
        title: "Bank Negara Malaysia maintains OPR at 3.00% in September meeting",
        body: "Kuala Lumpur: Bank Negara Malaysia decided to maintain the Overnight Policy Rate (OPR) at 3.00 percent on 7 September 2026. The central bank stated that monetary policy remains supportive of sustainable economic growth in Malaysia."
      },
      2: {
        title: "Maybank Chairman Tan Sri Dato' Seri Zamzamzamin addresses AGM",
        body: "Maybank Group held its annual gathering in Menara Maybank. Tan Sri Dato' Seri Zamzamzamin expressed confidence in the group's performance for Q3 2026, alongside Group CEO Datuk Khairussaleh Ramli."
      },
      3: {
        title: "S&P Global affirms rating for CIMB Group Berhad",
        body: "Rating agency S&P Global announced an upgraded credit outlook for CIMB Group Berhad and RHB Bank Berhad on 15 September 2026 in Singapore."
      },
      4: {
        title: "Maybank sumbang dana pemerkasaan usahawan di Kelantan",
        body: "Kota Bharu: Maybank memperuntukkan pembiayaan mikro bagi usahawan di Kelantan. Majlis penyerahan disempurnakan pada 20 September 2026."
      },
      5: {
        title: "Tips keselamatan perbankan digital untuk pengguna harian",
        body: "Sentiasa pastikan kata laluan akaun anda dirahsiakan dan jangan sekali-kali berkongsi kod pengesahan satu kali dengan mana-mana pihak. Kemas kini aplikasi telefon pintar anda secara berkala untuk perlindungan optimum."
      }
    };

    function loadSample(id) {
      const s = SAMPLES[id];
      if (s) {
        document.getElementById('titleInput').value = s.title;
        document.getElementById('bodyInput').value = s.body;
        document.getElementById('errorAlert').style.display = 'none';
      }
    }

    async function handleExtract(event) {
      event.preventDefault();

      const title = document.getElementById('titleInput').value;
      const body = document.getElementById('bodyInput').value;
      const errorAlert = document.getElementById('errorAlert');
      const submitBtn = document.getElementById('submitBtn');
      const btnSpinner = document.getElementById('btnSpinner');
      const btnText = document.getElementById('btnText');

      // Validasi sisi klien
      if (!title.trim() && !body.trim()) {
        errorAlert.textContent = "Judul dan isi artikel tidak boleh keduanya kosong.";
        errorAlert.style.display = 'block';
        return;
      }

      errorAlert.style.display = 'none';
      submitBtn.disabled = true;
      btnSpinner.style.display = 'inline-block';
      btnText.textContent = 'Menganalisis Entitas...';

      try {
        const response = await fetch('/extract-entities', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ title, body })
        });

        const data = await response.json();

        if (!response.ok) {
          throw new Error(data.message || 'Terjadi kesalahan pada layanan NER.');
        }

        renderResults(data, title, body);
      } catch (err) {
        errorAlert.textContent = err.message || "Gagal menghubungi server.";
        errorAlert.style.display = 'block';
      } finally {
        submitBtn.disabled = false;
        btnSpinner.style.display = 'none';
        btnText.textContent = 'Ekstrak Named Entity';
      }
    }

    function renderResults(data, title, body) {
      document.getElementById('emptyState').style.display = 'none';
      document.getElementById('statsBar').style.display = 'flex';
      document.getElementById('tableSection').style.display = 'block';

      const entities = data.entities || [];
      document.getElementById('totalEntCount').textContent = entities.length;
      document.getElementById('textLengthCount').textContent = data.meta ? data.meta.analyzed_length : (title.length + body.length + 2);
      document.getElementById('modelBadge').textContent = 'Model: ' + (data.meta ? data.meta.model : 'ner_model');

      const tbody = document.getElementById('entitiesTableBody');
      tbody.innerHTML = '';

      if (entities.length === 0) {
        tbody.innerHTML = '<tr><td colspan="4" style="text-align: center; color: var(--text-muted); padding: 16px;">Tidak ada named entity terdeteksi (Contoh Negatif).</td></tr>';
      } else {
        entities.forEach(ent => {
          const row = document.createElement('tr');
          row.innerHTML = `
            <td><strong>${escapeHtml(ent.text)}</strong></td>
            <td><span class="tag tag-${ent.label}">${ent.label}</span></td>
            <td style="font-family: 'Fira Code', monospace; color: var(--text-muted);">${ent.start}</td>
            <td style="font-family: 'Fira Code', monospace; color: var(--text-muted);">${ent.end}</td>
          `;
          tbody.appendChild(row);
        });
      }

      // Visual Highlight
      const analyzedText = title + "\\n\\n" + body;
      renderHighlight(analyzedText, entities);
    }

    function renderHighlight(text, entities) {
      const container = document.getElementById('highlightBox');
      if (!entities || entities.length === 0) {
        container.textContent = text;
        return;
      }

      // Urutkan span
      const sorted = [...entities].sort((a, b) => a.start - b.start);
      let html = '';
      let lastIndex = 0;

      sorted.forEach(ent => {
        if (ent.start >= lastIndex && ent.start < text.length) {
          html += escapeHtml(text.substring(lastIndex, ent.start));
          const entText = text.substring(ent.start, Math.min(ent.end, text.length));
          html += `<span class="hl-span hl-${ent.label}" data-label="${ent.label}">${escapeHtml(entText)}</span>`;
          lastIndex = ent.end;
        }
      });

      if (lastIndex < text.length) {
        html += escapeHtml(text.substring(lastIndex));
      }

      container.innerHTML = html;
    }

    function escapeHtml(str) {
      return str
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#039;');
    }
  </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def index_page() -> str:
    """Tampilkan Web UI Panel untuk ekstraksi entitas artikel."""
    return INDEX_HTML


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.api:app", host="127.0.0.1", port=8000, reload=True)
