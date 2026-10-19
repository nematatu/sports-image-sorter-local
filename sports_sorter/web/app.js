// Vanilla review UI. No build step, no external requests.
const state = {
  page: 1,
  pageSize: 60,
  label: "",
  only: "",
  engine: "",
  q: "",
  selected: new Set(),
  labels: [],
  items: [],
  total: 0,
};

const $ = (id) => document.getElementById(id);

function toast(message) {
  const el = $("toast");
  el.textContent = message;
  el.hidden = false;
  clearTimeout(el._t);
  el._t = setTimeout(() => { el.hidden = true; }, 2600);
}

async function api(path, options) {
  const res = await fetch(path, options);
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || res.statusText);
  }
  return res.json();
}

function labelName(id) {
  const found = state.labels.find((l) => l.id === id);
  return found ? found.name_ja : (id || "未分類");
}

function badgeClass(id) {
  if (id === "unclassified") return "badge unclassified";
  if (id === "other") return "badge other";
  return "badge";
}

async function loadSummary() {
  const s = await api("/api/summary");
  state.labels = s.labels;
  const parts = [
    `<span>合計 <b>${s.total}</b></span>`,
    `<span>レビュー済み <b>${s.reviewed}</b></span>`,
    `<span>未分類 <b>${s.counts.unclassified || 0}</b></span>`,
  ];
  $("summary").innerHTML = parts.join("");
  fillLabelSelects();
}

function fillLabelSelects() {
  const filter = $("label-filter");
  const bulk = $("bulk-label");
  const keepFilter = filter.value;
  const keepBulk = bulk.value;
  filter.innerHTML = '<option value="">すべての競技</option>';
  bulk.innerHTML = "";
  for (const lab of state.labels) {
    const o1 = document.createElement("option");
    o1.value = lab.id; o1.textContent = lab.name_ja;
    filter.appendChild(o1);
    const o2 = document.createElement("option");
    o2.value = lab.id; o2.textContent = lab.name_ja;
    bulk.appendChild(o2);
  }
  filter.value = keepFilter;
  bulk.value = keepBulk;
}

async function loadEngines() {
  const s = await api("/api/summary");
  const sel = $("engine-filter");
  sel.innerHTML = '<option value="">表示: 最終判定</option>';
  for (const engine of s.engines) {
    if (engine === "fusion") continue;
    const o = document.createElement("option");
    o.value = engine; o.textContent = `表示: ${engine}`;
    sel.appendChild(o);
  }
}

async function load() {
  const params = new URLSearchParams({
    page: String(state.page),
    page_size: String(state.pageSize),
  });
  if (state.label) params.set("label", state.label);
  if (state.only) params.set("only", state.only);
  if (state.engine) params.set("engine", state.engine);
  if (state.q) params.set("q", state.q);
  const data = await api(`/api/state?${params}`);
  state.items = data.items;
  state.total = data.total;
  render();
}

function render() {
  const grid = $("grid");
  grid.innerHTML = "";
  const fragment = document.createDocumentFragment();
  const votesVisible = (item) => Object.entries(item.votes).map(([engine, v]) => {
    if (!v || !v.label) return "";
    return `<span class="badge" title="${engine}">${engine}: ${labelName(v.label)}</span>`;
  }).join("");

  for (const item of state.items) {
    const card = document.createElement("article");
    card.className = "card" + (state.selected.has(item.id) ? " selected" : "");
    card.dataset.id = item.id;
    card.innerHTML = `
      <label class="checkbox"><input type="checkbox" ${state.selected.has(item.id) ? "checked" : ""} /></label>
      ${item.disagreement ? '<span class="flag">不一致</span>' : ""}
      <img class="thumb" loading="lazy" src="${item.thumb}" alt="" />
      <div class="body">
        <div class="name" title="${escapeHtml(item.rel_path)}">${escapeHtml(item.rel_path)}</div>
        <div class="row">
          <span class="${badgeClass(item.label)}">${labelName(item.label)}</span>
          ${item.source === "human" ? '<span class="badge human">human</span>' : ""}
          ${item.confidence != null ? `<span class="conf">${(item.confidence * 100).toFixed(0)}%</span>` : ""}
          ${item.error ? '<span class="badge warn">error</span>' : ""}
        </div>
        <div class="row">${votesVisible(item)}</div>
      </div>`;
    const cb = card.querySelector("input[type=checkbox]");
    cb.addEventListener("click", (ev) => ev.stopPropagation());
    cb.addEventListener("change", () => {
      if (cb.checked) state.selected.add(item.id); else state.selected.delete(item.id);
      card.classList.toggle("selected", cb.checked);
      updateSelection();
    });
    card.addEventListener("click", () => openDetail(item));
    fragment.appendChild(card);
  }
  grid.appendChild(fragment);
  $("empty").hidden = state.items.length > 0;
  const pages = Math.max(1, Math.ceil(state.total / state.pageSize));
  $("page-info").textContent = `${state.page} / ${pages} ページ（${state.total} 件）`;
  $("prev").disabled = state.page <= 1;
  $("next").disabled = state.page >= pages;
  updateSelection();
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

function updateSelection() {
  $("selection-count").textContent = `${state.selected.size} 件選択`;
  $("bulk-apply").disabled = state.selected.size === 0;
  $("bulk-clear").disabled = state.selected.size === 0;
}

async function openDetail(item) {
  const dialog = $("detail");
  $("detail-name").textContent = item.rel_path;
  $("detail-meta").textContent = `${labelName(item.label)}` +
    (item.confidence != null ? ` ・ 信頼度 ${(item.confidence * 100).toFixed(0)}%` : "") +
    ` ・ ${item.width}×${item.height}`;
  $("detail-img").src = `/api/full/${item.id}`;
  const votes = $("detail-votes");
  votes.innerHTML = "";
  for (const [engine, v] of Object.entries(item.votes)) {
    if (!v || !v.label) continue;
    const div = document.createElement("div");
    div.className = "vote";
    const conf = v.confidence != null ? `${(v.confidence * 100).toFixed(0)}%` : "-";
    div.innerHTML = `<span class="engine">${escapeHtml(engine)}</span>
      <span><b>${labelName(v.label)}</b> <span class="muted">${conf}</span></span>`;
    votes.appendChild(div);
  }
  if (!votes.children.length) votes.innerHTML = '<p class="muted">モデル判定なし</p>';

  const row = $("detail-labels");
  row.innerHTML = "";
  for (const lab of state.labels) {
    const button = document.createElement("button");
    button.textContent = lab.name_ja;
    button.className = item.human === lab.id || (!item.human && item.label === lab.id) ? "current" : "";
    button.addEventListener("click", async () => {
      await api("/api/label", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image_id: item.id, label_id: lab.id }),
      });
      toast(`${lab.name_ja} に修正しました`);
      dialog.close();
      await loadSummary();
      await load();
    });
    row.appendChild(button);
  }
  dialog.showModal();
}

// -- events -----------------------------------------------------------------
$("search").addEventListener("input", debounce(() => {
  state.q = $("search").value.trim();
  state.page = 1;
  load();
}, 250));

$("label-filter").addEventListener("change", () => {
  state.label = $("label-filter").value;
  state.page = 1;
  load();
});
$("only-filter").addEventListener("change", () => {
  state.only = $("only-filter").value;
  state.page = 1;
  load();
});
$("engine-filter").addEventListener("change", () => {
  state.engine = $("engine-filter").value;
  state.page = 1;
  load();
});
$("page-size").addEventListener("change", () => {
  state.pageSize = parseInt($("page-size").value, 10);
  state.page = 1;
  load();
});
$("reload").addEventListener("click", async () => { await loadSummary(); await load(); });
$("prev").addEventListener("click", () => { if (state.page > 1) { state.page--; load(); } });
$("next").addEventListener("click", () => { state.page++; load(); });
$("select-all").addEventListener("change", (ev) => {
  if (ev.target.checked) state.items.forEach((i) => state.selected.add(i.id));
  else state.items.forEach((i) => state.selected.delete(i.id));
  render();
});
$("bulk-clear").addEventListener("click", () => {
  state.selected.clear();
  $("select-all").checked = false;
  render();
});
$("bulk-apply").addEventListener("click", async () => {
  const label = $("bulk-label").value;
  if (!label || !state.selected.size) return;
  const ids = [...state.selected];
  const res = await api("/api/bulk", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ image_ids: ids, label_id: label }),
  });
  toast(`${res.updated} 件を ${labelName(label)} に修正しました`);
  state.selected.clear();
  $("select-all").checked = false;
  await loadSummary();
  await load();
});
$("detail-close").addEventListener("click", () => $("detail").close());
$("detail").addEventListener("click", (ev) => { if (ev.target === $("detail")) $("detail").close(); });
document.addEventListener("keydown", (ev) => { if (ev.key === "Escape") $("detail").close(); });

function debounce(fn, ms) {
  let timer;
  return (...args) => { clearTimeout(timer); timer = setTimeout(() => fn(...args), ms); };
}

(async function init() {
  try {
    await loadSummary();
    await loadEngines();
    await load();
  } catch (err) {
    document.body.innerHTML = `<pre style="padding:24px;color:#f87171">Failed to load review UI:\n${escapeHtml(err.message)}</pre>`;
  }
})();
