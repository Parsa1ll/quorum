const SVGNS = "http://www.w3.org/2000/svg";
const TOL = 1e-4;

let DATA = window.TTC_DATA || null;

if (DATA) {
  init();
} else {
  document.querySelector(".hero .lede").textContent =
    "Could not load data.js. Run `python eval/export_web.py` to generate it.";
}

// --- strategy math (mirrors src/strategies.py + eval/gsm8k.py) ---

function agreement(answers) {
  const xs = answers.filter((a) => a !== null);
  if (!xs.length) return 0;
  const counts = tally(xs);
  const top = Math.max(...counts.values());
  return top / xs.length;
}

function majorityVote(answers) {
  const xs = answers.filter((a) => a !== null);
  if (!xs.length) return null;
  const counts = tally(xs);
  let best = null, bestN = -1;
  for (const [v, n] of counts) if (n > bestN) { best = v; bestN = n; }
  return best;
}

function tally(xs) {
  const m = new Map();
  for (const x of xs) {
    const k = Math.round(x * 1e4) / 1e4;
    m.set(k, (m.get(k) || 0) + 1);
  }
  return m;
}

function runAdaptive(threshold) {
  const initN = DATA.meta.init_n, maxN = DATA.meta.k;
  const rows = DATA.pool.map((q, i) => {
    const early = q.answers.slice(0, initN);
    const used = agreement(early) >= threshold ? initN : maxN;
    const pred = majorityVote(q.answers.slice(0, used));
    const ok = pred !== null && Math.abs(pred - q.gold) < TOL;
    return { i, used, ok, pred, gold: q.gold, ag: agreement(early) };
  });
  const n = rows.length;
  return {
    rows,
    acc: rows.filter((r) => r.ok).length / n,
    avgSamples: rows.reduce((s, r) => s + r.used, 0) / n,
  };
}

function fixedPoint(name) {
  return DATA.fixed.find((f) => f.strategy === name);
}

// --- render ---

function init() {
  renderFixedStats();
  renderRules();
  renderOverthinking();
  setupReveal();

  const slider = document.getElementById("threshold");
  const out = document.getElementById("threshold-val");
  const update = () => {
    out.textContent = Number(slider.value).toFixed(2);
    renderAllocator(Number(slider.value));
  };
  slider.addEventListener("input", update);
  update();

  const url = repoUrl();
  if (url) {
    document.getElementById("repo-link").href = url;
    document.getElementById("footer-repo").href = url;
  }
}

function renderFixedStats() {
  const order = ["greedy", "sc@8", "sc@16"];
  const labels = { greedy: "Greedy (1 sample)", "sc@8": "Vote of 8", "sc@16": "Vote of 16" };
  const el = document.getElementById("fixed-stats");
  el.innerHTML = "";
  for (const name of order) {
    const f = fixedPoint(name);
    if (!f) continue;
    el.appendChild(card("stat", `
      <div class="num">${(f.accuracy * 100).toFixed(0)}%</div>
      <div class="lbl">${labels[name]}</div>
      <div class="sub">${f.avg_samples} sample${f.avg_samples === 1 ? "" : "s"} / question</div>`));
  }
}

function renderAllocator(threshold) {
  const res = runAdaptive(threshold);
  const sc16 = fixedPoint("sc@16");
  const delta = res.acc - sc16.accuracy;

  const metrics = document.getElementById("live-metrics");
  metrics.innerHTML = "";
  metrics.appendChild(card("metric", `
    <div class="num">${(res.acc * 100).toFixed(0)}%</div>
    <div class="lbl">Accuracy</div>
    <div class="delta ${Math.abs(delta) < 0.005 ? "flat" : "good"}">
      ${delta >= 0 ? "+" : ""}${(delta * 100).toFixed(1)} pts vs vote-of-16</div>`));
  metrics.appendChild(card("metric", `
    <div class="num">${res.avgSamples.toFixed(1)}</div>
    <div class="lbl">Avg samples / question</div>
    <div class="delta good">${(16 - res.avgSamples).toFixed(1)} fewer than vote-of-16</div>`));
  metrics.appendChild(card("metric", `
    <div class="num">${(100 * res.avgSamples / 16).toFixed(0)}%</div>
    <div class="lbl">Compute vs vote-of-16</div>
    <div class="delta flat">same accuracy, less work</div>`));

  const tiles = document.getElementById("tiles");
  tiles.innerHTML = "";
  for (const r of res.rows) {
    const t = document.createElement("div");
    t.className = "tile " + (r.ok ? "ok" : "bad");
    t.textContent = r.used;
    t.title = `Q${r.i + 1} · early agreement ${r.ag.toFixed(2)} · ` +
      `${r.used} samples · answered ${fmt(r.pred)} / correct ${fmt(r.gold)}`;
    tiles.appendChild(t);
  }

  renderScatter(res);
}

function renderScatter(res) {
  const svg = document.getElementById("scatter");
  clear(svg);
  const W = 600, H = 320, padL = 56, padR = 20, padT = 20, padB = 46;
  const xw = W - padL - padR, yh = H - padT - padB;
  const xdom = [0, 16], ydom = [0.6, 0.92];
  const mx = (v) => padL + ((v - xdom[0]) / (xdom[1] - xdom[0])) * xw;
  const my = (v) => padT + (1 - (v - ydom[0]) / (ydom[1] - ydom[0])) * yh;

  axis(svg, mx, my, xdom, ydom, [0, 4, 8, 12, 16], [0.6, 0.7, 0.8, 0.9],
    "samples per question", (v) => v, (v) => (v * 100).toFixed(0) + "%");

  for (const f of DATA.fixed) {
    circle(svg, mx(f.avg_samples), my(f.accuracy), 6, "dot");
    text(svg, mx(f.avg_samples) + 9, my(f.accuracy) + 4, f.strategy, "dotlabel");
  }
  // live adaptive point as a star
  star(svg, mx(res.avgSamples), my(res.acc), 9);
  text(svg, mx(res.avgSamples) + 11, my(res.acc) + 4, "adaptive", "dotlabel");
}

function renderOverthinking() {
  const svg = document.getElementById("overthinking");
  if (!DATA.overthinking.length) return;
  clear(svg);
  const pts = [...DATA.overthinking].sort((a, b) => a.avg_tokens - b.avg_tokens);
  const W = 600, H = 320, padL = 56, padR = 20, padT = 20, padB = 46;
  const xw = W - padL - padR, yh = H - padT - padB;
  const xmax = Math.ceil(Math.max(...pts.map((p) => p.avg_tokens)) / 100) * 100;
  const xdom = [0, xmax], ydom = [0.2, 0.9];
  const mx = (v) => padL + ((v - xdom[0]) / (xdom[1] - xdom[0])) * xw;
  const my = (v) => padT + (1 - (v - ydom[0]) / (ydom[1] - ydom[0])) * yh;

  const xticks = [];
  for (let t = 0; t <= xmax; t += Math.max(100, Math.round(xmax / 6 / 100) * 100)) xticks.push(t);
  axis(svg, mx, my, xdom, ydom, xticks, [0.2, 0.4, 0.6, 0.8],
    "avg thinking tokens", (v) => v, (v) => (v * 100).toFixed(0) + "%");

  let dpath = "";
  pts.forEach((p, i) => { dpath += (i ? "L" : "M") + mx(p.avg_tokens) + " " + my(p.accuracy) + " "; });
  const path = document.createElementNS(SVGNS, "path");
  path.setAttribute("d", dpath);
  path.setAttribute("class", "curve");
  svg.appendChild(path);
  for (const p of pts) circle(svg, mx(p.avg_tokens), my(p.accuracy), 4, "curvedot");
}

function renderRules() {
  const el = document.getElementById("rules");
  if (!DATA.stopping_rules.length) return;
  const ours = DATA.stopping_rules.find((r) => r.strategy === "ours(t=0.6)");
  const beta = DATA.stopping_rules.find((r) => r.strategy === "beta(c=0.8)");
  const sc16 = fixedPoint("sc@16");
  const rows = [
    { name: "Fixed vote of 16", acc: sc16.accuracy, samples: 16, win: false },
    { name: "My agreement rule", acc: ours.accuracy, samples: ours.avg_samples, win: false },
    { name: "Beta posterior (prior work)", acc: beta.accuracy, samples: beta.avg_samples, win: true },
  ];
  el.innerHTML = "";
  for (const r of rows) {
    const d = document.createElement("div");
    d.className = "rule" + (r.win ? " win" : "");
    d.innerHTML = `<span class="name">${r.name}</span>
      <span class="val">${(r.acc * 100).toFixed(0)}% acc</span>
      <span class="val">${r.samples.toFixed(2)} samples</span>`;
    el.appendChild(d);
  }
}

// --- svg helpers ---

function axis(svg, mx, my, xdom, ydom, xticks, yticks, xlabel, xfmt, yfmt) {
  line(svg, mx(xdom[0]), my(ydom[0]), mx(xdom[1]), my(ydom[0]));
  line(svg, mx(xdom[0]), my(ydom[0]), mx(xdom[0]), my(ydom[1]));
  for (const t of xticks) text(svg, mx(t), my(ydom[0]) + 22, xfmt(t), "axlabel", "middle");
  for (const t of yticks) text(svg, mx(xdom[0]) - 12, my(t) + 4, yfmt(t), "axlabel", "end");
  text(svg, (mx(xdom[0]) + mx(xdom[1])) / 2, my(ydom[0]) + 40, xlabel, "axlabel", "middle");
}

function line(svg, x1, y1, x2, y2) {
  const l = document.createElementNS(SVGNS, "line");
  l.setAttribute("x1", x1); l.setAttribute("y1", y1);
  l.setAttribute("x2", x2); l.setAttribute("y2", y2);
  l.setAttribute("class", "ax");
  svg.appendChild(l);
}

function circle(svg, cx, cy, r, cls) {
  const c = document.createElementNS(SVGNS, "circle");
  c.setAttribute("cx", cx); c.setAttribute("cy", cy); c.setAttribute("r", r);
  c.setAttribute("class", cls);
  svg.appendChild(c);
}

function star(svg, cx, cy, r) {
  let d = "";
  for (let i = 0; i < 10; i++) {
    const ang = (Math.PI / 5) * i - Math.PI / 2;
    const rad = i % 2 ? r * 0.45 : r;
    d += (i ? "L" : "M") + (cx + rad * Math.cos(ang)) + " " + (cy + rad * Math.sin(ang)) + " ";
  }
  const p = document.createElementNS(SVGNS, "path");
  p.setAttribute("d", d + "Z"); p.setAttribute("class", "star");
  svg.appendChild(p);
}

function text(svg, x, y, str, cls, anchor) {
  const t = document.createElementNS(SVGNS, "text");
  t.setAttribute("x", x); t.setAttribute("y", y);
  t.setAttribute("class", cls);
  if (anchor) t.setAttribute("text-anchor", anchor);
  t.textContent = str;
  svg.appendChild(t);
}

function clear(svg) { while (svg.firstChild) svg.removeChild(svg.firstChild); }

// gentle fade-in as each section scrolls in; no-JS keeps everything visible
function setupReveal() {
  if (!("IntersectionObserver" in window)) return;
  const els = document.querySelectorAll("main > section");
  els.forEach((el) => el.classList.add("reveal"));
  const io = new IntersectionObserver((entries) => {
    for (const e of entries) {
      if (e.isIntersecting) { e.target.classList.add("in"); io.unobserve(e.target); }
    }
  }, { threshold: 0.12 });
  els.forEach((el) => io.observe(el));
}

function card(cls, html) {
  const d = document.createElement("div");
  d.className = cls;
  d.innerHTML = html;
  return d;
}

function fmt(x) { return x === null ? "n/a" : (Number.isInteger(x) ? x : x.toFixed(2)); }

// Leave the repo link as-is unless a real URL is set here.
function repoUrl() { return null; }
