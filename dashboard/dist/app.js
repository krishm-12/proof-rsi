"use strict";

const $ = id => document.getElementById(id);
const text = (id, value) => { $(id).textContent = value; };
const number = (value, digits = 0) => Number.isFinite(value)
  ? value.toLocaleString(undefined, { maximumFractionDigits: digits }) : "—";
const percent = value => Number.isFinite(value) ? number(value * 100, 1) + "%" : "—";
const empty = message => el("div", "empty", message);
const nice = value => String(value ?? "").replaceAll("_", " ");
const duration = seconds => Number.isFinite(seconds)
  ? [Math.floor(seconds / 3600), Math.floor(seconds / 60) % 60, Math.floor(seconds) % 60]
      .map(value => String(value).padStart(2, "0")).join(":") : "—";
const kind = row => row.verification?.ok ? "verified" : row.interrupted ? "interrupted" : "rejected";
const stageName = value => nice(value).replace(/^round (\d+)/, (_, n) => "Round " + Number(n));
const PAGE_SIZE = 7;

function el(tag, className = "", content) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (content !== undefined) node.textContent = content;
  return node;
}

function statusTag(value, tone) {
  return el("span", "tag " + tone, value.toUpperCase());
}

function bar(parts) {
  const track = el("div", "bar");
  track.setAttribute("aria-hidden", "true");
  for (const [fraction, color] of parts) {
    const segment = el("i", color);
    segment.style.width = (fraction * 100) + "%";
    track.append(segment);
  }
  return track;
}

function options(select, first, values) {
  const chosen = select.value;
  const signature = JSON.stringify(values);
  if (select.dataset.options === signature) return;
  select.dataset.options = signature;
  select.replaceChildren(new Option(first, ""), ...values.map(([id, label]) => new Option(label, id)));
  if (values.some(([id]) => String(id) === chosen)) select.value = chosen;
}

const state = {
  snapshot: null, run: null, stamp: null, page: 0, proofIndex: null, detailIndex: null,
  proofSignature: null, cursor: -1, chart: null, inFlight: false, refreshQueued: false
};

async function refresh() {
  if (state.inFlight) {
    state.refreshQueued = true;
    return;
  }
  state.inFlight = true;
  const chosen = $("run").value;
  try {
    const response = await fetch("/api/snapshot" + (chosen ? "?run=" + encodeURIComponent(chosen) : ""), { cache: "no-store" });
    if (!response.ok) throw new Error("HTTP " + response.status);
    const data = await response.json();
    $("feed").classList.remove("error");
    text("feed", "●  Auto-refresh");
    if (chosen !== $("run").value) return;
    options($("run"), "Latest run", data.runs.map(run => [run.id, run.id]));
    const changedRun = data.selected?.id !== state.run;
    if (changedRun) {
      state.run = data.selected?.id;
      state.page = 0;
      state.proofIndex = null;
      state.detailIndex = null;
      state.proofSignature = null;
      $("loss-round").value = "";
    }
    const stamp = [data.selected?.updated, data.attempts.length, data.loss.length].join("/");
    state.snapshot = data;
    if (changedRun || stamp !== state.stamp) {
      state.stamp = stamp;
      renderExperiment(data);
    }
    renderHardware(data);
  } catch (error) {
    $("feed").classList.add("error");
    text("feed", "●  Feed unavailable");
    $("feed").title = String(error);
  } finally {
    state.inFlight = false;
    if (state.refreshQueued) {
      state.refreshQueued = false;
      refresh();
    }
  }
}

function renderExperiment({ selected, metrics: m, attempts, loss }) {
  const rounds = m.rounds ?? [];
  const baseline = m.baseline;
  const latest = rounds.filter(round => round.validation).at(-1);
  const score = latest?.validation ?? baseline;
  const outcomes = outcomeGroups(attempts, m.evidence?.attempt_outcomes);
  const totals = [...outcomes.values()];
  const verified = totals.reduce((sum, counts) => sum + counts.verified, 0);
  const rejected = totals.reduce((sum, counts) => sum + counts.rejected, 0);
  const generated = m.evidence?.attempts_total ?? attempts.length;
  const accepted = rounds.filter(round => round.accepted).length;
  const model = m.config?.model?.split(/[\\/]/).at(-1);
  const device = m.device ?? m.config?.device;
  const terminal = /^(complete|failed|interrupted|budget exhausted|smoke verified and trained)/i.test(m.status ?? "")
    && (m.status !== "smoke verified and trained" || m.config?.smoke_only);
  text("status", m.status ? nice(m.status).replace(/^./, letter => letter.toUpperCase()) : "Awaiting run data");
  text("updated", selected
    ? selected.id + " · " + (m.evidence ? "Archived evidence" : new Date(selected.updated * 1000).toLocaleTimeString()) : "No run selected");
  text("model", [model, device?.toUpperCase(), m.lean?.match(/Lean \(version ([^,]+)/)?.[1]?.replace(/^/, "Lean ")].filter(Boolean).join("  ·  ") || "Experiment monitor");
  text("generation", selected ? number(generated) : "—");
  text("generation-note", Number.isFinite(m.tokens?.generated) ? number(m.tokens.generated) + " total generated tokens" : "Logged proof attempts");
  text("verified", selected ? number(verified) : "—");
  text("verified-note", selected ? number(rejected) + " rejected attempts" : "Lean kernel checks");
  text("accuracy", percent(score?.accuracy));
  text("accuracy-note", score ? (latest ? "Round " + latest.round : "Baseline") + " · " + score.solved + "/" + score.tested + " solved" : "Awaiting validation");
  text("elapsed", duration(m.elapsed_s));
  text("budget", Number.isFinite(m.config?.minutes) ? number(m.config.minutes, 1) + " min configured budget" : "Budget unavailable");
  text("round-label", rounds.length ? "Round " + rounds.at(-1).round + (m.config?.rounds ? " / " + m.config.rounds : "") : "");
  text("pipeline-tag", terminal ? "STOPPED" : m.status ? "TRACKING" : "STANDBY");
  $("pipeline-tag").classList.toggle("live", Boolean(m.status && !terminal));
  const status = m.status ?? "";
  const phase = terminal ? "" : /training/.test(status) ? "train" : /validation|final/.test(status) ? "evaluate"
    : /collection|smoke/.test(status) ? "generate" : "";
  document.querySelectorAll("#pipeline li").forEach(node => {
    node.classList.toggle("active", node.dataset.phase === phase || (phase === "generate" && node.dataset.phase === "verify"));
  });
  text("pipe-generate", selected ? number(generated) + " logged attempts" : "Qwen continuation");
  text("pipe-verify", Number.isFinite(m.lean_subprocesses) ? number(m.lean_subprocesses) + " Lean subprocesses" : "Kernel and axiom audit");
  text("pipe-train", (m.loss?.length || loss.length) ? (loss.length || m.loss.length) + " logged optimizer steps" : "QLoRA optimizer");
  text("pipe-evaluate", accepted ? accepted + " accepted " + (accepted === 1 ? "adapter" : "adapters") : "Held-out validation");
  const stages = [["Base", baseline], ...rounds.map(round => ["R" + round.round, round.validation]),
    ["Final base", m.final?.base], ["Final adapter", m.final?.selected]].filter(([, summary]) => summary?.domains);
  renderMatrix("heatmap", stages, summary => summary.domains, value => [
    value.solved + "/" + value.tested, value.tested ? value.solved / value.tested : 0,
    value.solved + " Lean-verified of " + value.tested + " tested"
  ], "No domain evaluation recorded.");
  renderMatrix("difficulty", rounds.map(round => ["R" + round.round, round.levels]).filter(([, levels]) => levels),
    levels => levels, value => ["L" + value, value / 2, "Curriculum level " + value], "No curriculum rounds recorded.");
  renderOutcomes(outcomes);
  renderComparison(m, latest);
  renderCheckpoints(m, rounds);
  renderLoss();
  renderProofs();
}

function renderMatrix(id, columns, mapping, describe, message) {
  const domains = [...new Set(columns.flatMap(([, value]) => Object.keys(mapping(value))))];
  if (!domains.length) {
    $(id).replaceChildren(empty(message));
    return;
  }
  const table = el("table");
  const head = table.createTHead().insertRow();
  for (const label of ["Domain", ...columns.map(([label]) => label)]) {
    const heading = el("th", "", label);
    heading.scope = "col";
    head.append(heading);
  }
  const body = table.createTBody();
  for (const domain of domains) {
    const row = body.insertRow();
    const heading = el("th", "", domain);
    heading.scope = "row";
    row.append(heading);
    for (const [, summary] of columns) {
      const cell = row.insertCell();
      const value = mapping(summary)[domain];
      if (value == null) {
        cell.textContent = "—";
        continue;
      }
      const [label, intensity, description] = describe(value);
      cell.textContent = label;
      cell.dataset.value = label;
      cell.style.setProperty("--intensity", intensity);
      cell.title = description;
    }
  }
  $(id).replaceChildren(table);
}

function outcomeGroups(attempts, recorded) {
  if (recorded) return new Map(Object.entries(recorded));
  const groups = new Map();
  for (const row of attempts) {
    const key = row.attempt || "unknown";
    if (!groups.has(key)) groups.set(key, { verified: 0, rejected: 0, interrupted: 0 });
    groups.get(key)[kind(row)]++;
  }
  return groups;
}

function renderOutcomes(groups) {
  if (!groups.size) {
    $("outcomes").replaceChildren(empty("No proof attempts logged."));
    return;
  }
  const palette = [["verified", "", "var(--cyan)"], ["rejected", "red", "var(--red)"], ["interrupted", "amber", "var(--amber)"]];
  const nodes = Array.from(groups, ([label, counts]) => {
    const total = counts.verified + counts.rejected + counts.interrupted;
    const row = el("div", "outcome");
    row.title = palette.map(([name]) => counts[name] + " " + name).join(" · ");
    row.append(el("span", "", nice(label)), bar(palette.map(([name, color]) => [counts[name] / total, color])),
      el("span", "", counts.verified + " / " + counts.rejected + " / " + counts.interrupted));
    return row;
  });
  const legend = el("div", "legend");
  for (const [label, , color] of palette) {
    const item = el("span");
    const swatch = el("i");
    swatch.style.background = color;
    item.append(swatch, label);
    legend.append(item);
  }
  $("outcomes").replaceChildren(...nodes, legend);
}

function renderComparison(m, latest) {
  const heldOut = Boolean(m.final?.base);
  const base = heldOut ? m.final.base : m.baseline;
  const fine = heldOut ? m.final.selected : latest?.validation;
  text("comparison-label", heldOut ? "Final held-out benchmark · same theorem bank"
    : latest ? "Validation bank · candidate from round " + latest.round : "Validation bank");
  const grid = el("div", "compare");
  for (const [label, score, color] of [["Original Qwen", base, ""], ["Fine-tuned Qwen", fine, "violet"]]) {
    const box = el("div");
    box.append(el("span", "", label), el("strong", "", percent(score?.accuracy)),
      el("small", "", score ? score.solved + "/" + score.tested + " solved" + (score.complete ? "" : " · partial") : "No score recorded"));
    if (Number.isFinite(score?.accuracy)) box.append(bar([[score.accuracy, color]]));
    grid.append(box);
  }
  const paired = m.final?.paired;
  $("comparison").replaceChildren(grid);
  if (paired) $("comparison").append(el("div", "legend",
    number(paired.gains) + " gains · " + number(paired.losses) + " losses · exact p " + number(paired.two_sided_exact_p, 4)));
}

function renderCheckpoints(m, rounds) {
  if (!m.config && !rounds.length) {
    $("checkpoints").replaceChildren(empty("No run artifacts loaded."));
    return;
  }
  const flow = el("div", "checkpoints");
  const base = el("div", "checkpoint");
  base.append(el("strong", "", "Base model"), el("small", "", "Frozen backbone"));
  flow.append(base);
  for (const round of rounds) {
    const selected = Boolean(round.adapter && round.adapter === m.selected_adapter);
    const node = el("div", "checkpoint" + (selected ? " selected" : ""));
    const label = selected ? "Selected" : round.accepted ? "Accepted" : round.adapter && round.outcome ? "Reverted"
      : round.adapter ? "Evaluating" : round.outcome ? "No adapter" : "In progress";
    node.append(el("strong", "", "Round " + round.round), el("small", "", label));
    if (round.adapter) node.append(el("small", "", round.adapter));
    if (round.outcome) node.title = round.outcome;
    flow.append(node);
  }
  $("checkpoints").replaceChildren(flow);
}

function renderLoss() {
  const data = state.snapshot;
  if (!data) return;
  const rows = (data.loss.length ? data.loss : data.metrics.loss ?? []).filter(row => Number.isFinite(row.loss));
  const rounds = [...new Set(rows.map(row => row.round))].filter(Number.isFinite);
  options($("loss-round"), "All rounds", rounds.map(round => [round, "Round " + round]));
  const selected = $("loss-round").value;
  const points = rows.map((row, index) => ({ ...row, globalStep: index + 1 }))
    .filter(row => !selected || String(row.round) === selected);
  if (!points.length && Number.isFinite(data.metrics.smoke?.loss) && !selected) {
    points.push({ globalStep: 1, round: 0, step: 1, loss: data.metrics.smoke.loss, key: "smoke" });
  }
  state.chart = { points };
  state.cursor = -1;
  const smoke = points.length && points[0].key === "smoke";
  $("loss-empty").hidden = Boolean(points.length);
  text("loss-empty", "Waiting for logged optimizer steps");
  text("loss-footer", points.length
    ? number(points.length) + (smoke ? " smoke step" : " optimizer " + (points.length === 1 ? "step" : "steps")) + "  ·  Latest loss " + number(points.at(-1).loss, 5)
    : "No training trace recorded");
  drawLoss();
}

function drawLoss() {
  const canvas = $("loss-chart");
  const { width, height } = canvas.getBoundingClientRect();
  if (!width || !height) return;
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  canvas.width = Math.round(width * dpr);
  canvas.height = Math.round(height * dpr);
  const ctx = canvas.getContext("2d");
  ctx.scale(dpr, dpr);
  const tooltip = $("loss-tooltip");
  tooltip.hidden = true;
  const points = state.chart?.points ?? [];
  if (!points.length) return;
  const left = 42, right = width - 11, top = 13, bottom = height - 24;
  let low = Infinity, high = -Infinity;
  for (const point of points) {
    low = Math.min(low, point.loss);
    high = Math.max(high, point.loss);
  }
  const pad = Math.max((high - low) * .1, .02);
  const min = Math.max(0, low - pad), max = high + pad;
  const x = index => points.length === 1 ? (left + right) / 2 : left + index / (points.length - 1) * (right - left);
  const y = loss => bottom - (loss - min) / (max - min) * (bottom - top);
  state.chart.geometry = { left, right };
  ctx.font = "10px monospace";
  ctx.textAlign = "right";
  for (let tick = 0; tick <= 4; tick++) {
    const value = min + (max - min) * tick / 4;
    const line = y(value);
    ctx.strokeStyle = "#2c3550";
    ctx.beginPath();
    ctx.moveTo(left, line);
    ctx.lineTo(right, line);
    ctx.stroke();
    ctx.fillStyle = "#9aa5c2";
    ctx.fillText(number(value, 2), left - 6, line + 3);
  }
  ctx.textAlign = "center";
  const ticks = Math.min(points.length, Math.max(2, Math.floor((right - left) / 75)));
  for (let tick = 0; tick < ticks; tick++) {
    const index = ticks === 1 ? 0 : Math.round(tick * (points.length - 1) / (ticks - 1));
    ctx.fillText("Step " + points[index].globalStep, x(index), height - 5);
  }
  const gradient = ctx.createLinearGradient(0, top, 0, bottom);
  gradient.addColorStop(0, "#59e4ef36");
  gradient.addColorStop(1, "#59e4ef00");
  const path = () => points.forEach((point, index) => {
    ctx[index === 0 ? "moveTo" : "lineTo"](x(index), y(point.loss));
  });
  if (points.length > 1) {
    ctx.beginPath();
    path();
    ctx.lineTo(right, bottom);
    ctx.lineTo(left, bottom);
    ctx.closePath();
    ctx.fillStyle = gradient;
    ctx.fill();
    ctx.beginPath();
    path();
    ctx.strokeStyle = "#59e4ef";
    ctx.lineWidth = 2;
    ctx.stroke();
  }
  const index = state.cursor < 0 ? (points.length === 1 ? 0 : -1) : state.cursor;
  if (index < 0) return;
  const point = points[index];
  ctx.beginPath();
  ctx.arc(x(index), y(point.loss), 4, 0, Math.PI * 2);
  ctx.fillStyle = "#b39afa";
  ctx.fill();
  if (state.cursor < 0) return;
  tooltip.hidden = false;
  tooltip.textContent = (point.round === 0 ? "Smoke" : "Round " + point.round) + " · step " + point.step + " · loss " + number(point.loss, 5);
  tooltip.style.left = Math.max(0, Math.min(width - tooltip.offsetWidth, x(index) + 8)) + "px";
  tooltip.style.top = Math.max(0, y(point.loss) - 32) + "px";
  canvas.setAttribute("aria-label", tooltip.textContent + ". Use arrow keys to inspect optimizer steps.");
}

function renderProofs() {
  const entries = state.snapshot?.attempts ?? [];
  const query = $("search").value.trim().toLowerCase();
  const filter = $("proof-filter").value;
  const matches = [];
  for (let i = entries.length - 1; i >= 0; i--) {
    const row = entries[i];
    if (filter && kind(row) !== filter) continue;
    if (query && ![row.key, row.stage, row.domain, row.statement, row.raw].some(value => String(value ?? "").toLowerCase().includes(query))) continue;
    matches.push(i);
  }
  const pages = Math.ceil(matches.length / PAGE_SIZE);
  state.page = Math.max(0, Math.min(state.page, pages - 1));
  const visible = matches.slice(state.page * PAGE_SIZE, (state.page + 1) * PAGE_SIZE);
  const signature = JSON.stringify([visible, entries.length, state.proofIndex, query, filter]);
  if (state.proofSignature === signature) return;
  state.proofSignature = signature;
  if (!visible.includes(state.proofIndex)) state.proofIndex = visible[0] ?? null;
  text("proof-count", number(matches.length) + " / " + number(entries.length) + (state.snapshot?.metrics.evidence ? " sample attempts" : " attempts"));
  $("pagination").hidden = pages < 2;
  $("prev").disabled = state.page === 0;
  $("next").disabled = state.page + 1 >= pages;
  text("page", (state.page + 1) + " / " + pages);
  const nodes = visible.map(index => {
    const row = entries[index];
    const result = kind(row);
    const button = el("button", "proof-item");
    button.type = "button";
    button.setAttribute("aria-pressed", String(index === state.proofIndex));
    button.append(el("strong", "", row.key ?? "Untitled proof"),
      statusTag(result, result === "verified" ? "ok" : result === "interrupted" ? "warn" : "bad"),
      el("small", "", [stageName(row.stage), nice(row.attempt), row.domain].filter(Boolean).join("  ·  ")));
    button.addEventListener("click", () => showProof(index, button, nodes));
    return button;
  });
  $("proof-list").replaceChildren(...(nodes.length ? nodes : [empty(entries.length ? "No matching attempts." : "Waiting for proof attempts.")]));
  if (state.proofIndex === null) {
    $("proof-detail").replaceChildren(empty("Select an attempt to inspect its proof and Lean diagnostic."));
    return;
  }
  const row = entries[state.proofIndex];
  const previous = entries[state.detailIndex];
  if (!previous || previous.key !== row.key || previous.stage !== row.stage) state.detailIndex = state.proofIndex;
  renderAttempt(state.detailIndex);
}

function highlight(code) {
  const pre = el("pre");
  const tokens = String(code).split(/(\b(?:theorem|by|import|exact|apply|refine|intro|intros|assumption|rfl|cases|case|with|induction|have|show|calc|omega|simp|simpa|decide|constructor|rw|Nat|Int|Prop|List|sorry)\b|\b\d+\b|:=|=>|[∀∃≤≥≠¬∧∨↔→←⟨⟩=])/g);
  tokens.forEach((token, index) => {
    if (index % 2 === 0) pre.append(document.createTextNode(token));
    else pre.append(el("span", /^\d/.test(token) ? "lang-num" : /^[a-z]/i.test(token) ? "lang-kw" : "lang-op", token));
  });
  return pre;
}

function showProof(index, button, buttons) {
  state.proofIndex = index;
  state.detailIndex = index;
  buttons.forEach(node => node.setAttribute("aria-pressed", String(node === button)));
  renderAttempt(index);
}

function renderAttempt(index) {
  const attempts = state.snapshot.attempts;
  const row = attempts[index];
  if (!row) return;
  const result = kind(row);
  const detail = $("proof-detail");
  detail.replaceChildren(el("h3", "", row.key ?? "Untitled proof"),
    statusTag(result, result === "verified" ? "ok" : result === "interrupted" ? "warn" : "bad"));
  const timeline = el("div", "timeline");
  const attemptsByTheorem = attempts.map((attempt, position) => [attempt, position])
    .filter(([attempt]) => attempt.key === row.key && attempt.stage === row.stage);
  attemptsByTheorem.forEach(([attempt, position], ordinal) => {
    const button = el("button", "", (ordinal + 1) + " · " + nice(attempt.attempt));
    button.type = "button";
    button.setAttribute("aria-pressed", String(position === index));
    button.title = kind(attempt);
    button.addEventListener("click", () => {
      const scroll = detail.getBoundingClientRect().top;
      state.detailIndex = position;
      renderAttempt(position);
      timelineFocus(detail, ordinal, scroll);
    });
    timeline.append(button);
  });
  detail.append(timeline);
  if (row.statement) detail.append(el("h4", "", "Theorem"), highlight("theorem candidate " + row.statement + " := by"));
  detail.append(el("h4", "", "Generated proof · " + number(row.elapsed_s, 2) + " s · " + number(row.generated_tokens) + " tokens"),
    highlight(row.raw || "No proof text recorded."));
  const check = row.verification;
  if (check?.proof && String(check.proof).trim() !== String(row.raw ?? "").trim()) {
    detail.append(el("h4", "", "Lean-verified proof"), highlight(check.proof));
  }
  if (check?.diagnostic) detail.append(el("h4", "", "Lean diagnostic"), el("pre", "", check.diagnostic));
  if (check?.ok && Array.isArray(check.axioms)) {
    detail.append(el("h4", "", "Axiom audit"), el("pre", "", check.axioms.join(", ") || "No axioms reported."));
  }
}

function timelineFocus(detail, ordinal, previousTop) {
  detail.querySelectorAll(".timeline button")[ordinal]?.focus({ preventScroll: true });
  const shift = detail.getBoundingClientRect().top - previousTop;
  if (shift) window.scrollBy(0, shift);
}

function renderHardware({ gpu, metrics: m }) {
  const fields = gpu?.fields ?? {};
  const metric = (key, unit, digits = 0) => Number.isFinite(fields[key]) ? number(fields[key], digits) + unit : "—";
  const sampled = gpu?.sampled ? " · " + new Date(gpu.sampled * 1000).toLocaleTimeString() : "";
  text("gpu-name", gpu ? gpu.name + sampled : "GPU telemetry unavailable");
  const peak = m.gpu_peak?.allocated_mib;
  const cards = [
    ["GPU utilization", metric("utilization.gpu", "%"), "Device-wide sample"],
    ["VRAM used", metric("memory.used", " MiB"), Number.isFinite(fields["memory.total"]) ? number(fields["memory.total"]) + " MiB capacity" : "Device memory capacity unavailable"],
    ["GPU temperature", metric("temperature.gpu", " °C"), "Device temperature"],
    ["Board power", metric("power.draw", " W", 1), "Device power draw"]
  ].map(([label, value, description], index) => {
    const card = el("article", "card kpi");
    const header = el("div", "", label);
    header.append(el("span", "", ["◉", "▤", "♨", "ϟ"][index]));
    card.append(header, el("strong", "", value), el("small", "", description));
    return card;
  });
  $("hardware").replaceChildren(...cards);
  const measuredRounds = (m.rounds ?? []).filter(round => Number.isFinite(round.steps) && round.training_s > 0);
  const trainingTime = measuredRounds.reduce((sum, round) => sum + round.training_s, 0);
  const steps = measuredRounds.reduce((sum, round) => sum + round.steps, 0);
  const tokens = m.tokens ?? {};
  const usage = [
    ["Prompt tokens", number(tokens.prompt)], ["Generated tokens", number(tokens.generated)],
    ["Training sequence tokens", number(tokens.training_input)], ["Training answer tokens", number(tokens.training_answer)],
    ["Training throughput", trainingTime ? number(steps / trainingTime, 2) + " steps/s" : "—"],
    ["Run peak CUDA allocation", Number.isFinite(peak) ? number(peak, 1) + " MiB" : "—"]
  ].map(([label, value]) => {
    const box = el("div");
    box.append(el("span", "", label), el("strong", "", value));
    return box;
  });
  $("usage").replaceChildren(...usage);
}

$("run").addEventListener("change", refresh);
$("refresh").addEventListener("click", refresh);
$("loss-round").addEventListener("change", renderLoss);
for (const id of ["search", "proof-filter"]) $(id).addEventListener(id === "search" ? "input" : "change", () => {
  state.page = 0;
  state.proofIndex = null;
  renderProofs();
});
$("prev").addEventListener("click", () => { state.page--; renderProofs(); });
$("next").addEventListener("click", () => { state.page++; renderProofs(); });
const canvas = $("loss-chart");
canvas.addEventListener("pointermove", event => {
  const plot = state.chart;
  if (!plot?.geometry) return;
  const { left, right } = plot.geometry;
  const position = (event.offsetX - left) / (right - left);
  const index = Math.round(Math.max(0, Math.min(1, position)) * (plot.points.length - 1));
  if (state.cursor !== index) {
    state.cursor = index;
    drawLoss();
  }
});
canvas.addEventListener("pointerleave", () => { state.cursor = -1; drawLoss(); });
canvas.addEventListener("keydown", event => {
  if (!["ArrowLeft", "ArrowRight"].includes(event.key) || !state.chart?.points.length) return;
  event.preventDefault();
  state.cursor = Math.min(state.chart.points.length - 1, Math.max(0, state.cursor + (event.key === "ArrowLeft" ? -1 : 1)));
  drawLoss();
});
new ResizeObserver(drawLoss).observe($("loss-wrap"));
document.querySelector("nav").addEventListener("click", event => {
  const selected = event.target.closest("a");
  if (selected) document.querySelectorAll("nav a").forEach(link => link.classList.toggle("active", link === selected));
});
refresh();
window.setInterval(refresh, 4000);
