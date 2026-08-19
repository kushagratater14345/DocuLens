// DocuLens AI — frontend logic

const API = "";

const el = (id) => document.getElementById(id);

// ----------------------------------------------------------------------
// HEALTH CHECK
// ----------------------------------------------------------------------

async function checkHealth() {
  const pill = el("statusPill");
  const text = el("statusText");
  try {
    const res = await fetch(`${API}/health`);
    const data = await res.json();
    if (data.status === "ok") {
      pill.classList.remove("offline");
      text.textContent = "AI Engine Online";
    } else {
      pill.classList.add("offline");
      const issues = [];
      if (!data.ocr) issues.push("OCR");
      if (!data.database) issues.push("Database");
      if (!data.embedding_model) issues.push("Embeddings");
      text.textContent = `Degraded (${issues.join(", ")})`;
    }
  } catch (e) {
    pill.classList.add("offline");
    text.textContent = "Backend unreachable";
  }
}

// ----------------------------------------------------------------------
// UPLOAD
// ----------------------------------------------------------------------

const fileInput = el("fileInput");
const chooseBtn = el("chooseBtn");
const dropzone = el("dropzone");
const progressBox = el("progressBox");
const progressStep = el("progressStep");
const progressFill = el("progressFill");
const uploadError = el("uploadError");

chooseBtn.addEventListener("click", (e) => {
  e.preventDefault();
  fileInput.click();
});

fileInput.addEventListener("change", () => {
  if (fileInput.files.length > 0) {
    handleUpload(fileInput.files[0]);
  }
});

["dragover", "dragenter"].forEach((evt) =>
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropzone.classList.add("dragover");
  })
);
["dragleave", "drop"].forEach((evt) =>
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
  })
);
dropzone.addEventListener("drop", (e) => {
  const file = e.dataTransfer.files[0];
  if (file) handleUpload(file);
});

function setProgress(step, percent) {
  progressBox.style.display = "block";
  progressStep.textContent = step;
  progressFill.style.width = `${percent}%`;
}

async function handleUpload(file) {
  uploadError.style.display = "none";
  el("resultCard").style.display = "none";

  setProgress("Uploading…", 15);

  const formData = new FormData();
  formData.append("file", file);

  try {
    setProgress("Processing…", 35);

    const uploadPromise = fetch(`${API}/upload`, {
      method: "POST",
      body: formData,
    });

    setProgress("OCR complete", 55);
    await sleep(300);
    setProgress("Analyzing…", 70);

    const res = await uploadPromise;

    setProgress("Embedding…", 88);
    await sleep(200);

    const data = await res.json();

    if (!res.ok) {
      throw new Error(data.detail || "Upload failed.");
    }

    setProgress("Stored successfully", 100);
    await sleep(300);
    progressBox.style.display = "none";

    renderResult(data);
    loadStats();
    loadHistory();
  } catch (err) {
    progressBox.style.display = "none";
    uploadError.style.display = "block";
    uploadError.textContent = err.message || "Something went wrong while processing this document.";
  }
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

// ----------------------------------------------------------------------
// RESULT RENDERING
// ----------------------------------------------------------------------

function riskBadge(level) {
  const cls = level === "HIGH" ? "badge-high" : level === "MEDIUM" ? "badge-medium" : "badge-low";
  return `<span class="badge ${cls}">${level}</span>`;
}

function fmtCurrency(amount, currency) {
  if (amount === null || amount === undefined || amount === "") return "—";
  const symbol = currency === "USD" ? "$" : currency === "EUR" ? "€" : currency === "GBP" ? "£" : "₹";
  const num = Number(amount);
  return `${symbol}${num.toLocaleString(undefined, { maximumFractionDigits: 2 })}`;
}

function renderResult(data) {
  const card = el("resultCard");
  card.style.display = "block";
  card.scrollIntoView({ behavior: "smooth", block: "start" });

  const fields = data.fields || {};
  const currency = fields.currency || "INR";

  const displayFields = [
    ["Document Type", data.document_type],
    ["Confidence", `${Math.round(data.confidence * 100)}% (${data.confidence_status})`],
    ["Category", `<span class="category-badge">${data.category || "Other"}</span>`],
    ["Vendor", fields.vendor || fields.bank_name || fields.organization || "—"],
    ["Invoice Number", fields.invoice_number || "—"],
    ["Date", fields.date || "—"],
    ["Total", fmtCurrency(fields.total_amount ?? fields.closing_balance, currency)],
    ["GST", fmtCurrency(fields.gst, currency)],
    ["Risk", riskBadge(data.risk.risk_level)],
  ];

  el("resultGrid").innerHTML = displayFields
    .map(
      ([label, value]) => `
      <div class="result-field">
        <div class="result-field-label">${label}</div>
        <div class="result-field-value">${value}</div>
      </div>`
    )
    .join("");

  el("insightsList").innerHTML = (data.ai_insights || [])
    .map((insight) => `<li>${insight}</li>`)
    .join("");

  el("explainBox").innerHTML = renderExplainability(data);

  const related = data.related_documents || [];
  if (related.length > 0) {
    el("relatedSection").style.display = "block";
    el("relatedList").innerHTML = related
      .map((r) => renderDocRow(r.id, r.metadata, r.distance))
      .join("");
    attachDocRowHandlers(el("relatedList"));
  } else {
    el("relatedSection").style.display = "none";
  }

  el("extractedText").textContent = data.extracted_text || "";
}

function renderExplainability(data) {
  const breakdown = data.confidence_breakdown || {};
  const flagBreakdown = (data.risk && data.risk.flag_breakdown) || [];

  const rows = [
    ["Document classification", breakdown.classification_contribution || 0, 0.35],
    ["Field coverage", breakdown.field_coverage_contribution || 0, 0.5],
    ["OCR text quality", breakdown.ocr_quality_contribution || 0, 0.15],
  ];

  const confidenceHtml = rows
    .map(([label, value, max]) => {
      const pct = Math.min(100, (value / max) * 100);
      return `
      <div class="explain-row">
        <div class="explain-label">${label}</div>
        <div class="explain-bar-track"><div class="explain-bar-fill" style="width:${pct}%"></div></div>
        <div class="explain-value">+${value}</div>
      </div>`;
    })
    .join("");

  const noteHtml = breakdown.explanation
    ? `<div class="explain-note">${breakdown.explanation}</div>`
    : "";

  let flagsHtml = "";
  if (flagBreakdown.length > 0) {
    flagsHtml =
      `<div class="explain-note"><strong>Risk score breakdown:</strong></div>` +
      flagBreakdown
        .map(
          (f) => `
      <div class="explain-flag-row">
        <span>${f.flag}</span>
        <span class="explain-flag-points">+${f.points}</span>
      </div>`
        )
        .join("");
  }

  return confidenceHtml + noteHtml + flagsHtml;
}

// ----------------------------------------------------------------------
// STATS
// ----------------------------------------------------------------------

async function loadStats() {
  try {
    const res = await fetch(`${API}/stats`);
    const data = await res.json();

    el("statDocs").textContent = data.documents ?? 0;
    el("statSpending").textContent = fmtCurrency(data.total_spending, "INR");
    el("statGst").textContent = fmtCurrency(data.total_gst, "INR");
    el("statRisk").textContent = data.high_risk ?? 0;

    el("typeDistribution").innerHTML = renderDistribution(data.by_type || {}, false);
    el("categoryDistribution").innerHTML = renderDistribution(data.by_category || {}, true);

    renderTrendChart(data.spending_trend || []);
  } catch (e) {
    // Stats are non-critical; fail silently in the UI
  }
}

function renderDistribution(obj, isCurrency) {
  const entries = Object.entries(obj);
  if (entries.length === 0) return `<p class="muted">No documents yet.</p>`;

  const maxVal = Math.max(1, ...entries.map(([, v]) => v));
  return entries
    .sort((a, b) => b[1] - a[1])
    .map(([label, value]) => {
      const display = isCurrency ? fmtCurrency(value, "INR") : value;
      return `
      <div class="distribution-row">
        <div class="distribution-label">${label.replace(/_/g, " ")}</div>
        <div class="distribution-bar-track">
          <div class="distribution-bar-fill" style="width:${(value / maxVal) * 100}%"></div>
        </div>
        <div class="distribution-count">${display}</div>
      </div>`;
    })
    .join("");
}

function renderTrendChart(trend) {
  const container = el("trendChart");
  if (!trend || trend.length === 0) {
    container.innerHTML = `<div class="trend-empty">No spending data yet - upload a document to see the trend.</div>`;
    return;
  }

  const width = Math.max(420, trend.length * 60);
  const height = 160;
  const padding = 30;
  const maxAmount = Math.max(1, ...trend.map((t) => t.amount));
  const barWidth = (width - padding * 2) / trend.length - 10;

  const bars = trend
    .map((point, i) => {
      const barHeight = (point.amount / maxAmount) * (height - padding * 2);
      const x = padding + i * ((width - padding * 2) / trend.length);
      const y = height - padding - barHeight;
      const label = point.date.slice(5); // MM-DD
      return `
        <g>
          <rect class="trend-bar" x="${x}" y="${y}" width="${barWidth}" height="${barHeight}" rx="4">
            <title>${point.date}: ${fmtCurrency(point.amount, "INR")}</title>
          </rect>
          <text x="${x + barWidth / 2}" y="${height - padding + 16}" font-size="10" fill="#94a3b8" text-anchor="middle">${label}</text>
        </g>`;
    })
    .join("");

  container.innerHTML = `
    <svg viewBox="0 0 ${width} ${height + 10}" width="100%" height="${height + 10}">
      <defs>
        <linearGradient id="trendGradient" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stop-color="#6366f1" />
          <stop offset="100%" stop-color="#8b5cf6" />
        </linearGradient>
      </defs>
      ${bars}
    </svg>`;
}

// ----------------------------------------------------------------------
// HISTORY
// ----------------------------------------------------------------------

async function loadHistory() {
  const container = el("historyList");
  try {
    const res = await fetch(`${API}/documents`);
    const data = await res.json();
    const docs = data.documents || [];

    if (docs.length === 0) {
      container.innerHTML = `<p class="muted">No documents yet. Upload one to get started.</p>`;
      return;
    }

    container.innerHTML = docs.map((doc) => renderDocRow(doc.id, doc.metadata)).join("");
    attachDocRowHandlers(container, docs.map((d) => d.id));
  } catch (e) {
    container.innerHTML = `<p class="muted">Could not load document history.</p>`;
  }
}

function renderDocRow(id, meta, distance = null) {
  const title = meta.vendor || meta.bank_name || meta.organization || meta.source || "Untitled document";
  const sub = [meta.document_type, meta.invoice_number].filter(Boolean).join(" · ");
  const amount = meta.total_amount || meta.closing_balance;
  const currency = meta.currency || "INR";

  return `
    <div class="doc-row" data-id="${id}">
      <div class="doc-row-left">
        <div class="doc-row-title">${title}</div>
        <div class="doc-row-sub">${sub || "Document"} ${riskBadge(meta.risk_level || "LOW")}</div>
      </div>
      <div class="doc-row-right">
        <div class="doc-row-amount">${amount ? fmtCurrency(amount, currency) : ""}</div>
        ${distance !== null ? `<div class="doc-row-distance">distance: ${distance}</div>` : ""}
      </div>
    </div>`;
}

function attachDocRowHandlers(container) {
  container.querySelectorAll(".doc-row").forEach((row) => {
    row.addEventListener("click", () => openDocumentModal(row.dataset.id));
  });
}

// ----------------------------------------------------------------------
// SEARCH
// ----------------------------------------------------------------------

el("searchBtn").addEventListener("click", runSearch);
el("searchInput").addEventListener("keydown", (e) => {
  if (e.key === "Enter") runSearch();
});

async function runSearch() {
  const query = el("searchInput").value.trim();
  const resultsBox = el("searchResults");
  if (!query) return;

  resultsBox.innerHTML = `<p class="muted">Searching…</p>`;

  try {
    const res = await fetch(`${API}/search?q=${encodeURIComponent(query)}`);
    const data = await res.json();

    if (!res.ok) throw new Error(data.detail || "Search failed.");

    const results = data.results || [];
    if (results.length === 0) {
      resultsBox.innerHTML = `<p class="muted">No matching documents found.</p>`;
      return;
    }

    resultsBox.innerHTML = results
      .map((r, i) => `<div class="search-result-label muted">Result #${i + 1}</div>` + renderDocRow(r.id, r.metadata, r.distance))
      .join("");

    attachDocRowHandlers(resultsBox);
  } catch (err) {
    resultsBox.innerHTML = `<p class="muted">${err.message}</p>`;
  }
}

// ----------------------------------------------------------------------
// DOCUMENT DETAIL MODAL
// ----------------------------------------------------------------------

async function openDocumentModal(id) {
  try {
    const res = await fetch(`${API}/documents/${id}`);
    const doc = await res.json();
    if (!res.ok) throw new Error(doc.detail || "Could not load document.");

    const meta = doc.metadata;
    const currency = meta.currency || "INR";
    let flags = [];
    try {
      flags = JSON.parse((meta.risk_flags || "[]").replace(/'/g, '"'));
    } catch (e) {
      flags = [];
    }

    el("modalContent").innerHTML = `
      <h2>${meta.vendor || meta.bank_name || meta.organization || meta.source || "Document"}</h2>
      <p class="muted">${meta.source || ""}</p>
      <div class="result-grid" style="grid-template-columns: 1fr 1fr; margin-top:16px;">
        <div class="result-field"><div class="result-field-label">Type</div><div class="result-field-value">${meta.document_type}</div></div>
        <div class="result-field"><div class="result-field-label">Category</div><div class="result-field-value"><span class="category-badge">${meta.category || "Other"}</span></div></div>
        <div class="result-field"><div class="result-field-label">Confidence</div><div class="result-field-value">${Math.round((meta.confidence || 0) * 100)}%</div></div>
        <div class="result-field"><div class="result-field-label">Invoice Number</div><div class="result-field-value">${meta.invoice_number || "—"}</div></div>
        <div class="result-field"><div class="result-field-label">Date</div><div class="result-field-value">${meta.date || "—"}</div></div>
        <div class="result-field"><div class="result-field-label">Amount</div><div class="result-field-value">${fmtCurrency(meta.total_amount || meta.closing_balance, currency)}</div></div>
        <div class="result-field"><div class="result-field-label">GST</div><div class="result-field-value">${fmtCurrency(meta.gst, currency)}</div></div>
        <div class="result-field"><div class="result-field-label">Risk</div><div class="result-field-value">${riskBadge(meta.risk_level || "LOW")}</div></div>
      </div>
      ${flags.length ? `<h3>Risk Flags</h3><ul class="insights-list">${flags.map((f) => `<li>${f}</li>`).join("")}</ul>` : ""}
      <h3>Extracted Text</h3>
      <pre class="extracted-text">${doc.document || ""}</pre>
    `;

    el("modalBackdrop").style.display = "flex";
  } catch (err) {
    alert(err.message);
  }
}

el("modalClose").addEventListener("click", () => {
  el("modalBackdrop").style.display = "none";
});
el("modalBackdrop").addEventListener("click", (e) => {
  if (e.target.id === "modalBackdrop") el("modalBackdrop").style.display = "none";
});

// ----------------------------------------------------------------------
// ASK DOCULENS (RAG Q&A)
// ----------------------------------------------------------------------

el("askBtn").addEventListener("click", runAsk);
el("askInput").addEventListener("keydown", (e) => {
  if (e.key === "Enter") runAsk();
});

async function runAsk() {
  const question = el("askInput").value.trim();
  const answerBox = el("askAnswer");
  if (!question) return;

  answerBox.style.display = "block";
  answerBox.innerHTML = `<div class="ask-loading">Thinking… (first question loads a small local model, may take a moment)</div>`;

  try {
    const res = await fetch(`${API}/ask?q=${encodeURIComponent(question)}`);
    const data = await res.json();

    if (!res.ok) throw new Error(data.detail || "Couldn't get an answer.");

    const sources = data.sources || [];
    const sourceNames = sources
      .map((s) => s.metadata.vendor || s.metadata.bank_name || s.metadata.source || "document")
      .join(", ");

    answerBox.innerHTML = `
      <div class="ask-question">Q: ${data.question}</div>
      <div>${data.answer}</div>
      ${sourceNames ? `<div class="ask-sources">Based on: ${sourceNames}</div>` : ""}
    `;
  } catch (err) {
    answerBox.innerHTML = `<div class="ask-loading">${err.message}</div>`;
  }
}

// ----------------------------------------------------------------------
// VOICE SEARCH (Web Speech API - browser built-in, no backend needed)
// ----------------------------------------------------------------------

const voiceBtn = el("voiceBtn");
const SpeechRecognitionClass = window.SpeechRecognition || window.webkitSpeechRecognition;

if (!SpeechRecognitionClass) {
  voiceBtn.style.display = "none";
} else {
  const recognition = new SpeechRecognitionClass();
  recognition.lang = "en-US";
  recognition.interimResults = false;
  recognition.maxAlternatives = 1;

  let listening = false;

  voiceBtn.addEventListener("click", () => {
    if (listening) return;
    listening = true;
    voiceBtn.classList.add("recording");
    voiceBtn.textContent = "🔴";
    recognition.start();
  });

  recognition.addEventListener("result", (event) => {
    const transcript = event.results[0][0].transcript;
    el("searchInput").value = transcript;
    runSearch();
  });

  const resetVoiceBtn = () => {
    listening = false;
    voiceBtn.classList.remove("recording");
    voiceBtn.textContent = "🎤";
  };

  recognition.addEventListener("end", resetVoiceBtn);
  recognition.addEventListener("error", resetVoiceBtn);
}

// ----------------------------------------------------------------------
// INIT
// ----------------------------------------------------------------------

checkHealth();
loadStats();
loadHistory();
