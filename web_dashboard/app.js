const state = {
  weeks: [],
  selectedWeek: null,
  selectedYear: null,
  currentRunId: null,
  pollTimer: null,
  pdfRows: [],
  extractionRows: [],
  relevanceRows: [],
  matchingRows: [],
  reviewRows: [],
  matchTab: "final",
  activeQA: "pdfs",
  qaSaveTimers: new Map(),
  config: { review_only: false, can_start_pipeline: true },
};

const steps = [
  ["queued", "Lauf vorgemerkt", "Der Wochenlauf wird vorbereitet."],
  ["acquisition", "Prospekte sammeln", "PDFs der Wettbewerber werden heruntergeladen."],
  ["document_relevance", "Prospektrelevanz prüfen", "Nur lebensmittel- und markt-relevante PDFs werden behalten."],
  ["extraction", "Produkte extrahieren", "Produktnamen, Preise, Zeiträume und Quellen werden ausgelesen."],
  ["product_relevance", "Produktrelevanz prüfen", "Es wird geprüft, welche Produktzeilen relevant sind."],
  ["matching", "Produkte zuordnen", "Vergleichbare Produkte werden gruppiert und in Excel geschrieben."],
  ["report", "Berichte erstellen", "Die finalen Dateien werden fertiggestellt."],
  ["completed", "Abgeschlossen", "Der Wochenvergleich ist bereit."],
];

const statusLabels = {
  running: "läuft",
  completed: "abgeschlossen",
  failed: "fehlgeschlagen",
  queued: "vorgemerkt",
  idle: "bereit",
};

const messageTranslations = {
  "Pipeline is running.": "Pipeline läuft.",
  "Run queued.": "Lauf vorgemerkt.",
  "Pipeline completed.": "Pipeline abgeschlossen.",
  "Historic artifacts found on disk.": "Historische Dateien auf der Festplatte gefunden.",
};

const skipReasonLabels = {
  duplicate_previous_week: "Duplikat aus Vorwoche",
  irrelevant_non_food_only: "Nur Non-Food",
  irrelevant_market_scope: "Falscher Markt",
  irrelevant_marketing_or_magazine: "Marketing/Magazin statt Angebot",
  skipped: "Übersprungen",
};

const reviewIssueLabels = {
  "multiple products from same supplier": "Mehrere Produkte desselben Anbieters",
  "accepted close comparable match": "Nah vergleichbarer Match",
  "close comparable match needs review": "Nah vergleichbarer Match",
  "match confidence below exact threshold": "Match-Sicherheit unter Schwelle",
  "pair judge requested review": "KI fordert Prüfung an",
  "conflicting attributes present": "Auffällige Produktmerkmale",
  "attribute confidence below 85": "Produktmerkmale unsicher",
  "price spread above 80 percent": "Große Preisspanne",
  "quantity unit unknown": "Menge oder Einheit unklar",
};

const $ = (id) => document.getElementById(id);

document.addEventListener("DOMContentLoaded", async () => {
  bindNavigation();
  bindControls();
  initResizableQaLayouts();
  await loadConfig();
  applyAppMode();
  renderSteps({ step_index: -1 });
  await loadWeeks();
  await refreshCurrentView();
});

function bindNavigation() {
  document.querySelectorAll(".nav-item").forEach((button) => {
    button.addEventListener("click", async () => {
      document.querySelectorAll(".nav-item").forEach((item) => item.classList.remove("active"));
      button.classList.add("active");
      document.querySelectorAll(".view").forEach((view) => view.classList.remove("active"));
      $(`view-${button.dataset.view}`).classList.add("active");
      setTitle(button.dataset.view);
      await refreshCurrentView();
    });
  });
}

function bindControls() {
  $("refreshBtn").addEventListener("click", refreshAll);
  $("startRunBtn").addEventListener("click", startRun);
  ["pdfSearch", "pdfDecisionFilter", "pdfSkipReasonFilter", "pdfMistakeFilter", "pdfSupplier"].forEach((id) => $(id).addEventListener("input", renderPdfRelevance));
  ["extractionSearch", "extractionMistakeFilter", "extractionSupplier"].forEach((id) => $(id).addEventListener("input", renderExtraction));
  ["relevanceSearch", "relevanceFilter", "relevanceReasonFilter", "relevanceMistakeFilter", "relevanceSupplier"].forEach((id) => $(id).addEventListener("input", renderRelevance));
  ["matchingSearch", "matchingFilter", "matchingMistakeFilter", "matchingReasonFilter"].forEach((id) => $(id).addEventListener("input", renderMatching));
  document.querySelectorAll("[data-qa-view]").forEach((button) => {
    button.addEventListener("click", async () => {
      state.activeQA = button.dataset.qaView;
      switchQAView();
      await loadActiveQA();
    });
  });
  document.querySelectorAll(".segment").forEach((button) => {
    if (!button.dataset.matchTab) return;
    button.addEventListener("click", async () => {
      document.querySelectorAll("[data-match-tab]").forEach((item) => item.classList.remove("active"));
      button.classList.add("active");
      state.matchTab = button.dataset.matchTab;
      await loadMatching();
    });
  });
}

function initResizableQaLayouts() {
  document.querySelectorAll(".resizable-qa-layout").forEach((layout) => {
    const key = layout.dataset.resizeKey || "default";
    const storedWidth = localStorage.getItem(`qa-layout-width-${key}`);
    if (storedWidth) {
      layout.style.setProperty("--qa-left-width", storedWidth);
    }
    const handle = layout.querySelector(".qa-resizer");
    if (!handle) return;
    handle.addEventListener("pointerdown", (event) => {
      event.preventDefault();
      layout.classList.add("resizing");
      document.body.classList.add("qa-resizing");
      handle.setPointerCapture(event.pointerId);
      const onMove = (moveEvent) => resizeQaLayout(layout, key, moveEvent.clientX);
      const onEnd = () => {
        layout.classList.remove("resizing");
        document.body.classList.remove("qa-resizing");
        handle.removeEventListener("pointermove", onMove);
        handle.removeEventListener("pointerup", onEnd);
        handle.removeEventListener("pointercancel", onEnd);
      };
      handle.addEventListener("pointermove", onMove);
      handle.addEventListener("pointerup", onEnd);
      handle.addEventListener("pointercancel", onEnd);
    });
  });
}

function resizeQaLayout(layout, key, clientX) {
  const rect = layout.getBoundingClientRect();
  const minPane = Math.min(420, Math.max(240, rect.width * 0.22));
  const minLeft = minPane;
  const maxLeft = Math.max(minLeft, rect.width - minPane);
  const nextLeft = Math.min(maxLeft, Math.max(minLeft, clientX - rect.left));
  const percent = `${Math.round((nextLeft / rect.width) * 1000) / 10}%`;
  layout.style.setProperty("--qa-left-width", percent);
  localStorage.setItem(`qa-layout-width-${key}`, percent);
}

function setTitle(view) {
  const titles = {
    run: ["Pipeline starten", "Starte einen Wochenlauf und verfolge jeden wichtigen Schritt verständlich."],
    history: ["Historische Läufe", "Wähle eine abgeschlossene Woche aus und prüfe Extraktion, Relevanz und Produktzuordnung in einem Arbeitsbereich."],
  };
  $("pageTitle").textContent = titles[view][0];
  $("pageSubtitle").textContent = titles[view][1];
}

async function refreshAll() {
  await loadConfig();
  applyAppMode();
  await loadWeeks();
  await refreshCurrentView();
}

async function loadConfig() {
  try {
    state.config = await api("/api/config");
  } catch (error) {
    state.config = { review_only: false, can_start_pipeline: true };
  }
}

function applyAppMode() {
  const reviewOnly = Boolean(state.config?.review_only);
  document.body.classList.toggle("review-only", reviewOnly);
  const runNav = document.querySelector('[data-view="run"]');
  const historyNav = document.querySelector('[data-view="history"]');
  runNav?.classList.toggle("hidden", reviewOnly);
  $("view-run")?.classList.toggle("hidden", reviewOnly);
  $("startRunBtn").disabled = reviewOnly;
  if (reviewOnly) {
    runNav?.classList.remove("active");
    historyNav?.classList.add("active");
    document.querySelectorAll(".view").forEach((view) => view.classList.remove("active"));
    $("view-history").classList.add("active");
    $("serverStatus").textContent = "Kundenprüfung bereit";
    setTitle("history");
  } else {
    $("serverStatus").textContent = "Lokale Anwendung bereit";
  }
}

async function refreshCurrentView() {
  const active = document.querySelector(".nav-item.active")?.dataset.view || "run";
  if (active === "history") {
    await loadHistory();
    if (state.selectedWeek && state.selectedYear) {
      showSelectedInspector();
      await loadActiveQA();
    }
  }
}

async function loadWeeks() {
  const data = await api("/api/weeks");
  state.weeks = data.weeks || [];
  if (!$("runWeek").value && state.weeks.length) {
    $("runWeek").value = state.weeks[0].week;
    $("runYear").value = state.weeks[0].year;
  }
}

async function startRun() {
  if (state.config?.review_only || state.config?.can_start_pipeline === false) return;
  const week = Number($("runWeek").value);
  const year = Number($("runYear").value);
  $("startRunBtn").disabled = true;
  try {
    const run = await api("/api/runs", { method: "POST", body: JSON.stringify({ week, year }) });
    state.currentRunId = run.id;
    renderRun(run);
    startPolling();
  } finally {
    $("startRunBtn").disabled = false;
  }
}

function startPolling() {
  clearInterval(state.pollTimer);
  state.pollTimer = setInterval(pollRun, 1600);
  pollRun();
}

async function pollRun() {
  if (!state.currentRunId) return;
  const run = await api(`/api/runs/${state.currentRunId}`);
  renderRun(run);
  if (run.status === "completed" || run.status === "failed") {
    clearInterval(state.pollTimer);
    await loadWeeks();
  }
}

function renderRun(run) {
  $("runMessage").textContent = translateMessage(run.message || "Pipeline is running.");
  $("runBadge").textContent = statusLabel(run.status || "running");
  $("runBadge").className = `badge ${run.status || "running"}`;
  $("progressFill").style.width = `${run.progress || 0}%`;
  renderSteps(run);
  const logs = run.logs || [];
  $("runLogs").classList.toggle("empty-state", logs.length === 0);
  $("runLogs").innerHTML = logs.length
    ? logs.map((item) => `<div><span class="log-time">${escapeHtml((item.time || "").slice(11, 19))}</span> ${escapeHtml(translateMessage(item.message || ""))}</div>`).join("")
    : "Starte einen Lauf, um hier Echtzeit-Details zu sehen.";
  $("runLogs").scrollTop = $("runLogs").scrollHeight;
}

function renderSteps(run) {
  const current = Number(run.step_index ?? -1);
  $("stepList").innerHTML = steps
    .map((step, index) => {
      const klass = index < current ? "done" : index === current ? "active" : "";
      return `<div class="step ${klass}">
        <div class="step-title">${escapeHtml(step[1])}</div>
        <div class="step-copy">${escapeHtml(step[2])}</div>
      </div>`;
    })
    .join("");
}

async function loadHistory() {
  const data = await api("/api/runs");
  $("historyList").innerHTML = (data.runs || [])
    .map((run) => {
      const artifacts = run.artifacts || {};
      const selected = Number(run.week) === Number(state.selectedWeek) && Number(run.year) === Number(state.selectedYear);
      return `<button class="history-item ${selected ? "selected" : ""}" data-week="${run.week}" data-year="${run.year}" data-label="${escapeHtml(run.label || `KW${run.week}`)}">
        <div class="history-head">
          <div>
            <div class="cell-main">${escapeHtml(run.label || `KW${run.week}`)}</div>
            <div class="cell-muted">${escapeHtml(translateMessage(run.message || ""))}</div>
          </div>
          <span class="badge ${run.status || "completed"}">${escapeHtml(statusLabel(run.status || "completed"))}</span>
        </div>
        <div class="artifact-list">
          <span>Rohdaten: ${artifactLabel(Boolean(artifacts.raw_csv))}</span>
          <span>Relevanzdatei: ${artifactLabel(Boolean(artifacts.relevant_csv))}</span>
          <span>Produktzuordnung: ${artifactLabel(Boolean(artifacts.matching_xlsx))}</span>
        </div>
      </button>`;
    })
    .join("");
  document.querySelectorAll(".history-item").forEach((item) => {
    item.addEventListener("click", async () => {
      await selectHistoricRun(Number(item.dataset.week), Number(item.dataset.year));
    });
  });
  if (state.selectedWeek && state.selectedYear) {
    showSelectedInspector();
  }
}

async function selectHistoricRun(week, year) {
  state.selectedWeek = week;
  state.selectedYear = year;
  state.activeQA = "pdfs";
  await loadHistory();
  showSelectedInspector();
  switchQAView();
  await loadActiveQA();
}

function showSelectedInspector() {
  $("emptyInspector").classList.add("hidden");
  $("selectedInspector").classList.remove("hidden");
  $("selectedRunTitle").textContent = `KW${String(state.selectedWeek).padStart(2, "0")} ${state.selectedYear}`;
  $("selectedRunSubtitle").textContent = "Prüfe den ausgewählten Wochenlauf mit den Prüfwerkzeugen unten.";
}

function switchQAView() {
  document.querySelectorAll("[data-qa-view]").forEach((button) => {
    button.classList.toggle("active", button.dataset.qaView === state.activeQA);
  });
  document.querySelectorAll(".qa-view").forEach((view) => view.classList.remove("active"));
  $(`qa-${state.activeQA}`).classList.add("active");
}

async function loadActiveQA() {
  if (!state.selectedWeek || !state.selectedYear) return;
  if (state.activeQA === "pdfs") await loadPdfRelevance();
  if (state.activeQA === "extraction") await loadExtraction();
  if (state.activeQA === "relevance") await loadRelevance();
  if (state.activeQA === "matching") await loadMatching();
}

async function loadPdfRelevance() {
  const data = await api(`/api/pdf-relevance?week=${state.selectedWeek}&year=${state.selectedYear}`);
  state.pdfRows = data.rows || [];
  fillSupplierSelect("pdfSupplier", state.pdfRows);
  fillReasonSelect("pdfSkipReasonFilter", state.pdfRows, pdfSkipReasonValue, skipReasonLabel, "Alle Übersprunggründe");
  renderMetrics("pdfStats", [
    ["PDFs", data.stats?.total || 0],
    ["Relevant", data.stats?.relevant || 0],
    ["Übersprungen", data.stats?.skipped || 0],
  ]);
  renderPdfSkipBreakdown(data.stats?.skip_reasons || []);
  renderPdfRelevance();
}

function renderPdfSkipBreakdown(items) {
  if (!items.length) {
    $("pdfSkipBreakdown").innerHTML = "";
    return;
  }
  $("pdfSkipBreakdown").innerHTML = `<div class="breakdown-title">Übersprunggründe</div>
    <div class="breakdown-list">
      ${items.map((item) => `<div class="breakdown-item">
        <strong>${escapeHtml(item.count || 0)}</strong>
        <div>
          <div class="cell-main">${escapeHtml(skipReasonLabel(item.label || "skipped"))}</div>
          <div class="cell-muted">${escapeHtml(item.example || "")}</div>
        </div>
      </div>`).join("")}
    </div>`;
}

function renderPdfRelevance() {
  const search = $("pdfSearch").value.toLowerCase();
  const filter = $("pdfDecisionFilter").value;
  const skipReason = $("pdfSkipReasonFilter").value;
  const supplier = $("pdfSupplier").value;
  const rows = state.pdfRows.filter((row) => {
    const haystack = `${row.filename} ${row.title} ${row.supplier} ${row.relevance_label} ${row.relevance_reason} ${row.skip_label} ${skipReasonLabel(pdfSkipReasonValue(row))} ${row.skip_reason}`.toLowerCase();
    if (search && !haystack.includes(search)) return false;
    if (supplier && row.supplier !== supplier) return false;
    if (filter === "relevant" && !row.is_relevant) return false;
    if (filter === "skipped" && row.is_relevant) return false;
    if (skipReason && pdfSkipReasonValue(row) !== skipReason) return false;
    if (!passesMistakeFilter(row, "pdfMistakeFilter")) return false;
    return true;
  });
  $("pdfTable").innerHTML = table(
    ["Fehler", "Entscheidung", "PDF", "Übersprunggrund", "Gültigkeit"],
    rows.map((row) => [
      qaMarkCell(row),
      pdfDecisionCell(row),
      productCell(row.filename, [row.supplier, row.title || row.tab].filter(Boolean).join(" · ")),
      pdfSkipReasonCell(row),
      textCell([row.valid_from, row.valid_to].filter(Boolean).join(" bis "), marketLabel(row)),
    ]),
    rows.map((row) => () => selectPdfPreview(row))
  );
}

function selectPdfPreview(row) {
  $("pdfPreviewMeta").textContent = `${row.filename || "PDF"} · ${row.supplier || ""}`;
  const preview = row._preview || {};
  if (preview.type === "image") {
    $("pdfPreviewBox").innerHTML = `<img src="${preview.url}" alt="PDF-Relevanz-Vorschau" />`;
  } else if (preview.type === "pdf") {
    $("pdfPreviewBox").innerHTML = `<iframe src="${preview.url}" title="PDF-Dokument"></iframe>`;
  } else {
    $("pdfPreviewBox").innerHTML = "Keine PDF-Vorschau verfügbar.";
    $("pdfPreviewBox").classList.add("empty-state");
  }
}

function pdfDecisionCell(row) {
  const relevant = Boolean(row.is_relevant);
  const confidence = row.relevance_confidence ? `${Math.round(Number(row.relevance_confidence) * 100)}%` : "";
  return `<span class="pill ${relevant ? "ok" : "no"}">${relevant ? "Relevant" : "Übersprungen"}</span>
    <div class="cell-muted">${escapeHtml(confidence)}</div>`;
}

function pdfSkipReasonCell(row) {
  if (Boolean(row.is_relevant)) {
    return `<div class="cell-muted">-</div>`;
  }
  return textCell(skipReasonLabel(pdfSkipReasonValue(row)), row.skip_reason || row.relevance_reason || "");
}

function pdfSkipReasonValue(row) {
  if (Boolean(row.is_relevant)) return "";
  return row.skip_label || row.relevance_label || "skipped";
}

function skipReasonLabel(value) {
  const raw = String(value || "skipped");
  return skipReasonLabels[raw] || raw.replaceAll("_", " ");
}

function marketLabel(row) {
  if (row.market_scope === "all") return "Alle Märkte";
  if (Array.isArray(row.valid_markets)) return row.valid_markets.join(", ");
  return row.market_scope || "";
}

async function loadExtraction() {
  const data = await api(`/api/products?week=${state.selectedWeek}&year=${state.selectedYear}&view=extraction`);
  state.extractionRows = data.rows || [];
  fillSupplierSelect("extractionSupplier", state.extractionRows);
  renderMetrics("extractionStats", [
    ["Produkte", data.stats?.total || 0],
    ["Anbieter", data.stats?.suppliers?.length || 0],
  ]);
  renderExtraction();
}

function renderExtraction() {
  const search = $("extractionSearch").value.toLowerCase();
  const supplier = $("extractionSupplier").value;
  const rows = state.extractionRows.filter((row) => {
    const haystack = `${row.product_name} ${row.description} ${row.supplier} ${row.source_file}`.toLowerCase();
    return (!search || haystack.includes(search)) && (!supplier || row.supplier === supplier) && passesMistakeFilter(row, "extractionMistakeFilter");
  });
  $("extractionTable").innerHTML = table(
    ["Fehler", "Produkt", "Anbieter", "Preis", "Quelle"],
    rows.slice(0, 500).map((row) => [
      qaMarkCell(row),
      productCell(row.product_name, row.description),
      textCell(row.supplier, row.category),
      textCell(priceLabel(row), validLabel(row)),
      textCell(fileName(row.source_file), `Seite ${row.source_page || ""}`),
    ]),
    rows.slice(0, 500).map((row) => () => selectPreview(row))
  );
}

function selectPreview(row) {
  $("previewMeta").textContent = `${fileName(row.source_file)} Seite ${row.source_page || ""}`;
  const preview = row._preview || {};
  if (preview.type === "image") {
    $("previewBox").innerHTML = `<img src="${preview.url}" alt="Quellseite" />`;
  } else if (preview.type === "pdf") {
    $("previewBox").innerHTML = `<iframe src="${preview.url}" title="Quell-PDF"></iframe>`;
  } else {
    $("previewBox").innerHTML = "Keine Quellenvorschau verfügbar.";
    $("previewBox").classList.add("empty-state");
  }
}

async function loadRelevance() {
  const data = await api(`/api/products?week=${state.selectedWeek}&year=${state.selectedYear}&view=relevance`);
  state.relevanceRows = data.rows || [];
  fillSupplierSelect("relevanceSupplier", state.relevanceRows);
  fillReasonSelect("relevanceReasonFilter", state.relevanceRows, (row) => row.Reason || "", (value) => value || "Ohne Grund", "Alle Gründe");
  renderMetrics("relevanceStats", [
    ["Zeilen", data.stats?.total || 0],
    ["Relevant", data.stats?.relevant || 0],
    ["Zeitlich relevant", data.stats?.time_relevant || 0],
  ]);
  renderRelevance();
}

function renderRelevance() {
  const search = $("relevanceSearch").value.toLowerCase();
  const filter = $("relevanceFilter").value;
  const reason = $("relevanceReasonFilter").value;
  const supplier = $("relevanceSupplier").value;
  const rows = state.relevanceRows.filter((row) => {
    const haystack = `${row.product_name} ${row.product} ${row.brand} ${row.description} ${row.Reason}`.toLowerCase();
    if (search && !haystack.includes(search)) return false;
    if (supplier && row.supplier !== supplier) return false;
    if (reason && String(row.Reason || "") !== reason) return false;
    if (filter === "ja" && !isYes(row.Relevant)) return false;
    if (filter === "nein" && isYes(row.Relevant)) return false;
    if (filter === "time" && !isYes(row["Relevant Time"])) return false;
    if (!passesMistakeFilter(row, "relevanceMistakeFilter")) return false;
    return true;
  });
  $("relevanceTable").innerHTML = table(
    ["Fehler", "Entscheidung", "Produkt", "Grund", "Anbieter", "Preis"],
    rows.slice(0, 700).map((row) => [
      qaMarkCell(row),
      decisionCell(row.Relevant, row["Relevant Time"]),
      productCell(row.product || row.product_name, row.description),
      textCell(row.Reason || "", row.brand ? `Marke: ${row.brand}` : ""),
      textCell(row.supplier, row.category),
      textCell(priceLabel(row), validLabel(row)),
    ])
  );
}

async function loadMatching() {
  const sheet = state.matchTab === "short" ? "Final Output Short" : "Final Output";
  const data = await api(`/api/matching?week=${state.selectedWeek}&year=${state.selectedYear}&sheet=${encodeURIComponent(sheet)}`);
  state.matchingRows = data.rows || [];
  state.reviewRows = data.review || [];
  fillReasonSelect("matchingReasonFilter", state.reviewRows, (row) => row.issue_type || row.issue_summary || "Prüfung", reviewIssueLabel, "Alle Prüfgründe");
  renderMetrics("matchingStats", [
    ["Gruppen", data.stats?.total || 0],
    ["2+ Anbieter", data.stats?.matched_multi || 0],
    ["Einzelanbieter", data.stats?.single_supplier || 0],
    ["Prüfung", data.stats?.review || 0],
  ]);
  renderMatching();
}

function renderMatching() {
  $("matchingReasonFilter").classList.toggle("hidden", state.matchTab !== "review");
  if (state.matchTab === "review") {
    renderReview();
    return;
  }
  const search = $("matchingSearch").value.toLowerCase();
  const filter = $("matchingFilter").value;
  const rows = state.matchingRows.filter((row) => {
    const present = ["Metro", "Selgros", "Handelshof", "Edeka"].filter((key) => row[key]).length;
    if (filter === "multi" && present < 2) return false;
    if (filter === "single" && present > 1) return false;
    if (!passesMistakeFilter(row, "matchingMistakeFilter")) return false;
    const haystack = Object.values(row).join(" ").toLowerCase();
    return !search || haystack.includes(search);
  });
  $("matchingTable").innerHTML = table(
    ["Fehler", "Produkt", "Details", "Metro", "Selgros", "Handelshof", "Edeka"],
    rows.slice(0, 500).map((row) => [
      qaMarkCell(row),
      productCell(row.Produkt || row.Product, row.Kategorie || row.Category),
      textCell(row.Marke || "", row.Herkunft || ""),
      offerCell(row.Metro),
      offerCell(row.Selgros),
      offerCell(row.Handelshof),
      offerCell(row.Edeka),
    ])
  );
}

function renderReview() {
  const search = $("matchingSearch").value.toLowerCase();
  const reason = $("matchingReasonFilter").value;
  const rows = state.reviewRows.filter((row) => {
    if (!passesMistakeFilter(row, "matchingMistakeFilter")) return false;
    if (reason && String(row.issue_type || row.issue_summary || "Prüfung") !== reason) return false;
    return !search || Object.values(row).join(" ").toLowerCase().includes(search);
  });
  const intro = `<div class="review-explainer">
    <div>
      <strong>Was bedeutet diese Prüfung?</strong>
      <p>Hier landen Produktgruppen, bei denen die KI eine Zuordnung gefunden hat, aber ein Nutzer sie vor Preisentscheidungen kontrollieren sollte. Häufigster Fall: Ein Anbieter hat mehrere ähnliche Produkte in derselben Gruppe, zum Beispiel zwei Prospekte, Packungsgrößen oder Produktvarianten.</p>
    </div>
    <div>
      <strong>Was ist zu tun?</strong>
      <p>Prüfe, ob die betroffenen Produkte wirklich derselbe Artikel sind. Wenn nicht, ist die Gruppe ein Matching-Fehler und sollte markiert oder später getrennt werden.</p>
    </div>
  </div>`;
  $("matchingTable").innerHTML = intro + (rows.length
    ? table(
        ["Fehler", "Prüfgrund", "Produktgruppe", "Betroffene Produkte", "Anbieter & Preise", "KI-Hinweis", "Empfohlene Aktion"],
        rows.map((row) => [
          qaMarkCell(row),
          textCell(reviewIssueLabel(row.issue_type || row.issue_summary || "Prüfung"), row.canonical_product_id || ""),
          productCell(row.canonical_product_name || "", `${splitList(row.product_ids).length || splitList(row.product_names).length} Produktzeilen in dieser Gruppe`),
          offerCell(formatReviewProducts(row.product_names)),
          offerCell(formatReviewSuppliers(row.suppliers, row.prices)),
          offerCell(reviewReasonText(row)),
          textCell(reviewActionText(row), formatConflictAttributes(row.conflicting_attributes)),
        ])
      )
    : `<div class="empty-state review-empty">Keine Prüffälle für die aktuellen Filter. Das bedeutet: Für diese Ansicht gibt es keine Produktgruppen, die eine manuelle Kontrolle brauchen.</div>`);
}

function reviewIssueLabel(value) {
  const raw = String(value || "Prüfung").trim();
  return raw
    .split(";")
    .map((part) => part.trim())
    .filter(Boolean)
    .map((part) => reviewIssueLabels[part] || part.replaceAll("_", " "))
    .join(" · ");
}

function reviewActionText(row) {
  const issue = String(row.issue_type || row.issue_summary || "").toLowerCase();
  if (issue.includes("multiple products from same supplier")) {
    return "Innerhalb jedes Anbieters prüfen, ob es echte Dubletten oder akzeptable Varianten sind. Falls unterschiedliche Artikel zusammenstehen, als Fehler markieren.";
  }
  if (issue.includes("conflicting attributes")) {
    return "Auffällige Produktmerkmale vergleichen und nur freigeben, wenn es trotz Abweichung derselbe Artikel ist.";
  }
  return "Manuell prüfen, bevor diese Produktgruppe für Preisentscheidungen verwendet wird.";
}

function reviewReasonText(row) {
  const parts = [];
  if (row.issue_summary) parts.push(`Systemgrund: ${reviewIssueLabel(row.issue_summary)}`);
  if (row.llm_reason) parts.push(shortenText(row.llm_reason, 900));
  return parts.join("\n\n");
}

function formatReviewProducts(value) {
  const products = splitList(value);
  if (!products.length) return "";
  return products.map((product, index) => `${index + 1}. ${product}`).join("\n");
}

function formatReviewSuppliers(suppliersValue, pricesValue) {
  const suppliers = splitList(suppliersValue);
  const prices = splitList(pricesValue);
  if (!suppliers.length && !prices.length) return "";
  return [
    suppliers.length ? `Anbieter: ${suppliers.join(", ")}` : "",
    prices.length ? `Preise: ${prices.map(formatPriceText).join(", ")}` : "",
  ].filter(Boolean).join("\n");
}

function formatConflictAttributes(value) {
  const conflicts = splitList(String(value || "").replaceAll("|", ";"));
  if (!conflicts.length) return "";
  const labels = [...new Set(conflicts.map(germanAttributeLabel))];
  return `Auffällige Merkmale: ${labels.join(", ")}`;
}

function germanAttributeLabel(value) {
  const labels = {
    brand: "Marke",
    description: "Beschreibung",
    product_name: "Produktname",
    processing: "Verarbeitung",
    packaging: "Verpackung",
    quantity: "Menge",
    quantity_unit: "Mengeneinheit",
    quality_class: "Qualitätsklasse",
    calibre: "Kaliber",
    origin: "Herkunft",
    unit: "Einheit",
  };
  const key = String(value || "").trim();
  return labels[key] || key;
}

function splitList(value) {
  return String(value || "")
    .split(";")
    .map((item) => item.trim())
    .filter(Boolean);
}

function shortenText(value, maxLength = 600) {
  const text = String(value || "").trim();
  if (text.length <= maxLength) return text;
  return `${text.slice(0, maxLength).trim()} ...`;
}

function formatPriceText(value) {
  const text = String(value || "").trim();
  const number = Number(text.replace(",", "."));
  if (!Number.isNaN(number)) return `${number.toLocaleString("de-DE", { minimumFractionDigits: 2, maximumFractionDigits: 2 })} EUR`;
  return text;
}

function qaMarkCell(row) {
  if (!row._qa_key || !row._qa_scope) {
    return `<div class="cell-muted">-</div>`;
  }
  const mark = row._qa_mark || {};
  const checked = mark.is_mistake ? "checked" : "";
  return `<div class="qa-mark-cell" onclick="event.stopPropagation()">
    <label class="qa-check">
      <input type="checkbox" data-qa-check data-qa-scope="${escapeHtml(row._qa_scope)}" data-qa-key="${escapeHtml(row._qa_key)}" ${checked} />
      Fehler
    </label>
    <textarea data-qa-comment data-qa-scope="${escapeHtml(row._qa_scope)}" data-qa-key="${escapeHtml(row._qa_key)}" rows="2" placeholder="Kommentar">${escapeHtml(mark.comment || "")}</textarea>
    <div class="qa-save-status" data-qa-status data-qa-scope="${escapeHtml(row._qa_scope)}" data-qa-key="${escapeHtml(row._qa_key)}"></div>
  </div>`;
}

function passesMistakeFilter(row, filterId) {
  const filter = $(filterId)?.value || "all";
  const mark = row._qa_mark || {};
  if (filter === "mistakes") return Boolean(mark.is_mistake);
  if (filter === "comments") return Boolean(String(mark.comment || "").trim());
  return true;
}

function bindQaMarkInputs() {
  document.querySelectorAll("[data-qa-check]").forEach((input) => {
    if (input.dataset.bound) return;
    input.dataset.bound = "1";
    input.addEventListener("click", (event) => event.stopPropagation());
    input.addEventListener("change", () => saveQaMark(input.dataset.qaScope, input.dataset.qaKey));
  });
  document.querySelectorAll("[data-qa-comment]").forEach((input) => {
    if (input.dataset.bound) return;
    input.dataset.bound = "1";
    input.addEventListener("click", (event) => event.stopPropagation());
    input.addEventListener("input", () => {
      const timerKey = `${input.dataset.qaScope}:${input.dataset.qaKey}`;
      clearTimeout(state.qaSaveTimers.get(timerKey));
      state.qaSaveTimers.set(timerKey, setTimeout(() => saveQaMark(input.dataset.qaScope, input.dataset.qaKey), 650));
    });
    input.addEventListener("blur", () => saveQaMark(input.dataset.qaScope, input.dataset.qaKey));
  });
}

async function saveQaMark(scope, key) {
  if (!scope || !key || !state.selectedWeek || !state.selectedYear) return;
  const checkbox = document.querySelector(`[data-qa-check][data-qa-scope="${scope}"][data-qa-key="${key}"]`);
  const commentInput = document.querySelector(`[data-qa-comment][data-qa-scope="${scope}"][data-qa-key="${key}"]`);
  const status = document.querySelector(`[data-qa-status][data-qa-scope="${scope}"][data-qa-key="${key}"]`);
  if (status) status.textContent = "Speichert...";
  try {
    const result = await api("/api/qa-mark", {
      method: "POST",
      body: JSON.stringify({
        week: state.selectedWeek,
        year: state.selectedYear,
        scope,
        key,
        is_mistake: Boolean(checkbox?.checked),
        comment: commentInput?.value || "",
      }),
    });
    updateLocalQaMark(scope, key, result.mark || {});
    if (status) {
      status.textContent = "Gespeichert";
      setTimeout(() => {
        if (status.textContent === "Gespeichert") status.textContent = "";
      }, 1200);
    }
  } catch (error) {
    if (status) status.textContent = "Speichern fehlgeschlagen";
    console.error(error);
  }
}

function updateLocalQaMark(scope, key, mark) {
  [state.pdfRows, state.extractionRows, state.relevanceRows, state.matchingRows, state.reviewRows].forEach((rows) => {
    rows.forEach((row) => {
      if (row._qa_scope === scope && row._qa_key === key) {
        row._qa_mark = {
          is_mistake: Boolean(mark.is_mistake),
          comment: mark.comment || "",
          updated_at: mark.updated_at || "",
        };
      }
    });
  });
}

function table(headers, rows, clickHandlers = []) {
  const body = rows
    .map((cells, index) => `<tr data-row="${index}">${cells.map((cell) => `<td>${cell}</td>`).join("")}</tr>`)
    .join("");
  setTimeout(() => {
    document.querySelectorAll("tr[data-row]").forEach((row) => {
      const handler = clickHandlers[Number(row.dataset.row)];
      if (handler) {
        row.addEventListener("click", (event) => {
          if (event.target.closest(".qa-mark-cell")) return;
          document.querySelectorAll("tbody tr").forEach((item) => item.classList.remove("selected"));
          row.classList.add("selected");
          handler();
        });
      }
    });
    bindQaMarkInputs();
  }, 0);
  return `<table><thead><tr>${headers.map((header) => `<th>${escapeHtml(header)}</th>`).join("")}</tr></thead><tbody>${body}</tbody></table>`;
}

function productCell(main, sub) {
  return `<div class="cell-main">${escapeHtml(main || "")}</div><div class="cell-muted">${escapeHtml(sub || "")}</div>`;
}

function textCell(main, sub) {
  return `<div>${escapeHtml(main || "")}</div><div class="cell-muted">${escapeHtml(sub || "")}</div>`;
}

function offerCell(value) {
  return `<div class="offer-cell">${escapeHtml(value || "")}</div>`;
}

function decisionCell(relevant, timeRelevant) {
  const rel = isYes(relevant);
  const time = isYes(timeRelevant);
  return `<span class="pill ${rel ? "ok" : "no"}">${rel ? "Relevant" : "Nicht relevant"}</span>
    <div style="height:5px"></div>
    <span class="pill ${time ? "ok" : "warn"}">${time ? "Zeit ok" : "Zeit nein"}</span>`;
}

function renderMetrics(id, metrics) {
  $(id).innerHTML = metrics.map(([label, value]) => `<div class="metric"><strong>${value}</strong><span>${escapeHtml(label)}</span></div>`).join("");
}

function fillSupplierSelect(id, rows) {
  const current = $(id).value;
  const suppliers = [...new Set(rows.map((row) => row.supplier).filter(Boolean))].sort();
  $(id).innerHTML = `<option value="">Alle Anbieter</option>${suppliers.map((supplier) => `<option value="${supplier}">${supplier}</option>`).join("")}`;
  if (suppliers.includes(current)) $(id).value = current;
}

function fillReasonSelect(id, rows, valueGetter, labelGetter = (value) => value, defaultLabel = "Alle Gründe") {
  const element = $(id);
  if (!element) return;
  const current = element.value;
  const counts = new Map();
  rows.forEach((row) => {
    const value = String(valueGetter(row) || "").trim();
    if (!value) return;
    counts.set(value, (counts.get(value) || 0) + 1);
  });
  const options = [...counts.entries()].sort((a, b) => String(labelGetter(a[0])).localeCompare(String(labelGetter(b[0])), "de"));
  element.innerHTML = `<option value="">${escapeHtml(defaultLabel)}</option>${options
    .map(([value, count]) => `<option value="${escapeHtml(value)}">${escapeHtml(labelGetter(value))} (${count})</option>`)
    .join("")}`;
  if (counts.has(current)) element.value = current;
}

function priceLabel(row) {
  const price = row.price ? `${formatNumber(row.price)} EUR` : "";
  const perKg = row.price_per_kg ? `${formatNumber(row.price_per_kg)} EUR/kg` : "";
  return [price, perKg].filter(Boolean).join(" | ");
}

function validLabel(row) {
  return [row.valid_from, row.valid_to].filter(Boolean).join(" bis ");
}

function translateMessage(value) {
  const text = String(value || "");
  if (text.startsWith("Starting command:")) return text.replace("Starting command:", "Starte Befehl:");
  if (text.startsWith("Dashboard error:")) return text.replace("Dashboard error:", "Dashboard-Fehler:");
  if (text.startsWith("Pipeline failed with exit code")) return text.replace("Pipeline failed with exit code", "Pipeline fehlgeschlagen mit Exit-Code");
  return messageTranslations[text] || text;
}

function statusLabel(value) {
  return statusLabels[String(value || "")] || value || "";
}

function artifactLabel(available) {
  return available ? "vorhanden" : "fehlt";
}

function fileName(path) {
  return String(path || "").split("/").pop();
}

function isYes(value) {
  return ["ja", "yes", "true", "1", "x", "relevant"].includes(String(value || "").toLowerCase());
}

function formatNumber(value) {
  const number = Number(String(value).replace(",", "."));
  if (Number.isNaN(number)) return value;
  return number.toLocaleString("de-DE", { maximumFractionDigits: 2 });
}

async function api(url, options = {}) {
  const response = await fetch(url, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!response.ok) throw new Error(await response.text());
  return response.json();
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}
