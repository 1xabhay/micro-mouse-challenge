"use strict";

// Wall bits follow the .maz standard: N=1, E=2, S=4, W=8.
const N = 1, E = 2, S = 4, W = 8;
const HEADING_ANGLE = { N: -Math.PI / 2, E: 0, S: Math.PI / 2, W: Math.PI };

const state = {
  maze: null,
  result: null,
  frame: 0,
  playing: false,
  lastTick: 0,
  bots: [],
  race: null,
};

const $ = (id) => document.getElementById(id);
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const botColor = (name) => css(`--${name}`) || css("--floodfill");
const fmt = (v, digits = 2) =>
  v === null || v === undefined || !Number.isFinite(v) ? "—" : Number(v).toFixed(digits);

// An unsolved maze scores infinity, which JSON cannot carry — it arrives as
// null. Rank on the solved count instead of trusting the number.
const isRanked = (row) => row.solved === row.attempts && Number.isFinite(row.mean_score);

// ---------------------------------------------------------------- rendering

function drawMaze() {
  const canvas = $("maze");
  const ctx = canvas.getContext("2d");
  const maze = state.maze;
  if (!maze) return;

  // Redraw at device resolution so walls stay crisp on HiDPI screens.
  const dpr = window.devicePixelRatio || 1;
  const cssSize = canvas.clientWidth || 720;
  if (canvas.width !== Math.round(cssSize * dpr)) {
    canvas.width = canvas.height = Math.round(cssSize * dpr);
  }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

  const pad = 10;
  const span = cssSize - pad * 2;
  const cell = span / maze.size;
  // Maze y grows north, canvas y grows down, so flip on the way out.
  const px = (x) => pad + x * cell;
  const py = (y) => pad + (maze.size - 1 - y) * cell;

  ctx.clearRect(0, 0, cssSize, cssSize);
  ctx.fillStyle = css("--surface-1");
  ctx.fillRect(0, 0, cssSize, cssSize);

  // faint cell grid
  ctx.strokeStyle = css("--grid");
  ctx.lineWidth = 1;
  for (let i = 0; i <= maze.size; i++) {
    const p = pad + i * cell;
    ctx.beginPath(); ctx.moveTo(pad, p); ctx.lineTo(pad + span, p); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(p, pad); ctx.lineTo(p, pad + span); ctx.stroke();
  }

  // goal block + start cell
  ctx.fillStyle = css("--goal");
  for (const [gx, gy] of maze.goal) ctx.fillRect(px(gx), py(gy), cell, cell);
  ctx.fillStyle = css("--surface-2");
  ctx.fillRect(px(maze.start[0]), py(maze.start[1]), cell, cell);

  const upto = currentFrame();

  // cells the mouse has actually sensed
  if ($("showVisited").checked && state.result) {
    const seen = new Set();
    for (let i = 0; i <= upto; i++) {
      const p = state.result.trace[i];
      seen.add(p.x + "," + p.y);
    }
    ctx.fillStyle = botColor(state.result.bot) + "1f";
    for (const key of seen) {
      const [x, y] = key.split(",").map(Number);
      ctx.fillRect(px(x), py(y), cell, cell);
    }
  }

  // the true optimal route, for reference
  if ($("showOptimal").checked && maze.optimal_path.length) {
    ctx.strokeStyle = css("--text-muted");
    ctx.lineWidth = 2;
    ctx.setLineDash([4, 5]);
    ctx.beginPath();
    maze.optimal_path.forEach(([x, y], i) => {
      const cx = px(x) + cell / 2, cy = py(y) + cell / 2;
      i ? ctx.lineTo(cx, cy) : ctx.moveTo(cx, cy);
    });
    ctx.stroke();
    ctx.setLineDash([]);
  }

  // the mouse's own path
  if (state.result && upto > 0) {
    ctx.strokeStyle = botColor(state.result.bot);
    ctx.lineWidth = 2.5;
    ctx.lineJoin = ctx.lineCap = "round";
    ctx.beginPath();
    for (let i = 0; i <= upto; i++) {
      const p = state.result.trace[i];
      const cx = px(p.x) + cell / 2, cy = py(p.y) + cell / 2;
      i ? ctx.lineTo(cx, cy) : ctx.moveTo(cx, cy);
    }
    ctx.stroke();
  }

  // walls, drawn last so nothing paints over them
  ctx.strokeStyle = css("--wall");
  ctx.lineWidth = 3;
  ctx.lineCap = "square";
  for (let y = 0; y < maze.size; y++) {
    for (let x = 0; x < maze.size; x++) {
      const bits = maze.walls[y][x];
      const left = px(x), top = py(y), right = left + cell, bottom = top + cell;
      const line = (x1, y1, x2, y2) => {
        ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke();
      };
      if (bits & N) line(left, top, right, top);
      if (bits & S) line(left, bottom, right, bottom);
      if (bits & W) line(left, top, left, bottom);
      if (bits & E) line(right, top, right, bottom);
    }
  }

  // the mouse itself
  if (state.result) {
    const p = state.result.trace[upto];
    const cx = px(p.x) + cell / 2, cy = py(p.y) + cell / 2;
    const r = cell * 0.3;
    ctx.save();
    ctx.translate(cx, cy);
    ctx.rotate(HEADING_ANGLE[p.heading] ?? 0);
    ctx.fillStyle = botColor(state.result.bot);
    ctx.strokeStyle = css("--surface-1");
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(r, 0); ctx.lineTo(-r * 0.75, r * 0.75); ctx.lineTo(-r * 0.75, -r * 0.75);
    ctx.closePath();
    ctx.fill(); ctx.stroke();
    ctx.restore();
  }
}

const currentFrame = () =>
  state.result ? Math.min(state.frame, state.result.trace.length - 1) : 0;

// ---------------------------------------------------------------- playback

function tick(now) {
  requestAnimationFrame(tick);
  if (!state.playing || !state.result) return;

  const speed = parseFloat($("speed").value);
  const dt = (now - state.lastTick) / 1000;
  state.lastTick = now;

  const trace = state.result.trace;
  const target = trace[currentFrame()].t + dt * speed;
  while (state.frame < trace.length - 1 && trace[state.frame + 1].t <= target) {
    state.frame++;
  }
  if (state.frame >= trace.length - 1) setPlaying(false);

  $("scrub").value = state.frame;
  $("clock").textContent = fmt(trace[currentFrame()].t) + " s";
  drawMaze();
}

function setPlaying(on) {
  state.playing = on;
  $("play").textContent = on ? "Pause" : "Play";
  state.lastTick = performance.now();
}

// ------------------------------------------------------------------ panels

function renderTiles() {
  const r = state.result;
  if (!r) return;
  const tiles = [
    ["Score", fmt(r.score), "s"],
    ["Best run", fmt(r.best_run_time), "s"],
    ["Search", fmt(r.search_time), "s"],
    ["Maze time", fmt(r.maze_time, 1), "s"],
    ["Runs", r.runs.length, ""],
    ["Explored", (r.coverage * 100).toFixed(0), "%"],
  ];
  $("tiles").innerHTML = tiles
    .map(([k, v, u]) => `<div class="tile"><div class="k">${k}</div>
      <div class="v">${v}<span class="u">${u ? " " + u : ""}</span></div></div>`)
    .join("");
}

function renderRace(data) {
  state.race = data;
  const rows = data.leaderboard;
  const ranked = rows.filter(isRanked).map((r) => r.mean_score);
  const max = ranked.length ? Math.max(...ranked) : 1;

  $("legend").innerHTML = rows
    .map((r) => `<span><span class="swatch" style="background:${botColor(r.bot)}"></span>${r.bot}</span>`)
    .join("");

  // Horizontal bars: one categorical comparison of mean handicapped score.
  $("chart").innerHTML = rows
    .map((r) => {
      const ok = isRanked(r);
      // A bot that failed a maze has no finite score: show a full muted track
      // labelled "unsolved" rather than a short bar that would read as a win.
      const width = ok ? Math.max(2, (r.mean_score / max) * 100) : 100;
      return `<div class="bar-row">
        <div class="bar-label">${r.bot}</div>
        <div class="bar-track"><div class="bar-fill" style="width:${width}%;background:${
          ok ? botColor(r.bot) : "var(--surface-2)"
        }"></div></div>
        <div class="bar-val">${ok ? fmt(r.mean_score) + "s" : `${r.solved}/${r.attempts}`}</div>
      </div>`;
    })
    .join("");

  renderPerMaze(data);

  $("board").querySelector("tbody").innerHTML = rows
    .map(
      (r) => `<tr>
        <td><span class="swatch" style="background:${botColor(r.bot)}"></span>${r.bot}
            <span style="color:var(--text-muted)"> · ${r.style}</span></td>
        <td>${isRanked(r) ? fmt(r.mean_score) : "∞"}</td>
        <td>${r.solved}/${r.attempts}</td>
        <td>${r.wins}</td>
        <td>${fmt(r.mean_run)}</td>
        <td>${fmt(r.mean_search)}</td>
        <td>${(r.mean_coverage * 100).toFixed(0)}%</td>
      </tr>`
    )
    .join("");
}

function renderPerMaze(data) {
  // Column order follows the leaderboard ranking so the two tables agree.
  const bots = data.leaderboard.map((r) => r.bot);
  const byKey = new Map(data.attempts.map((a) => [`${a.seed}|${a.bot}`, a]));

  $("perMaze").querySelector("thead").innerHTML =
    "<tr><th>Maze</th>" +
    bots.map((b) => `<th><span class="swatch" style="background:${botColor(b)}"></span>${b}</th>`).join("") +
    "</tr>";

  $("perMaze").querySelector("tbody").innerHTML = data.seeds
    .map((seed) => {
      const cells = bots.map((b) => byKey.get(`${seed}|${b}`));
      const finite = cells.filter((c) => c && c.solved).map((c) => c.score);
      const best = finite.length ? Math.min(...finite) : null;
      return `<tr><td>${seed}</td>` + cells
        .map((c) => {
          if (!c || !c.solved) return `<td class="miss">∞</td>`;
          return `<td class="${c.score === best ? "best" : ""}">${fmt(c.score)}</td>`;
        })
        .join("") + "</tr>";
    })
    .join("");
}

// ------------------------------------------------------------------ actions

async function loadMaze() {
  const seed = parseInt($("seed").value, 10) || 0;
  state.maze = await (await fetch(`/api/maze?seed=${seed}`)).json();
  state.result = null;
  state.frame = 0;
  $("scrub").max = 0;
  $("runStatus").textContent = `Maze ${seed} · optimal route ${state.maze.optimal_cells} cells`;
  drawMaze();
}

async function runBot() {
  const bot = $("bot").value;
  const seed = parseInt($("seed").value, 10) || 0;
  $("run").disabled = true;
  $("runStatus").textContent = `Running ${bot}…`;
  try {
    const res = await fetch("/api/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ bot, seed }),
    });
    if (!res.ok) throw new Error((await res.json()).detail);
    const data = await res.json();
    state.maze = data.maze;
    state.result = data.result;
    state.frame = 0;
    $("scrub").max = data.result.trace.length - 1;
    $("runStatus").textContent = data.result.solved
      ? `${bot} solved maze ${seed} in ${data.result.steps} steps · ${data.result.runs.length} run(s)`
      : `${bot} did not reach the centre (${data.result.steps} steps)`;
    renderTiles();
    setPlaying(true);
  } catch (err) {
    $("runStatus").textContent = "Error: " + err.message;
  } finally {
    $("run").disabled = false;
  }
}

async function race() {
  const mazes = parseInt($("mazes").value, 10) || 5;
  const bots = state.bots.map((b) => b.name);
  $("race").disabled = true;
  $("raceStatus").textContent = `Racing ${bots.length} bots over ${mazes} mazes…`;
  try {
    const res = await fetch("/api/race", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ bots, mazes }),
    });
    const data = await res.json();
    renderRace(data);
    $("raceStatus").textContent =
      `${mazes} mazes · handicapped score = best run + maze time / 30. Lower is better.`;
  } catch (err) {
    $("raceStatus").textContent = "Error: " + err.message;
  } finally {
    $("race").disabled = false;
  }
}

// -------------------------------------------------------------------- setup

async function init() {
  state.bots = (await (await fetch("/api/bots")).json()).bots;
  $("bot").innerHTML = state.bots
    .map((b) => `<option value="${b.name}">${b.name} — ${b.style}</option>`)
    .join("");
  const describe = () => {
    const bot = state.bots.find((b) => b.name === $("bot").value);
    $("botDesc").textContent = bot ? bot.description : "";
  };
  if (state.bots.some((b) => b.name === "floodfill")) $("bot").value = "floodfill";
  describe();

  $("bot").addEventListener("change", describe);
  $("run").addEventListener("click", runBot);
  $("race").addEventListener("click", race);
  $("newMaze").addEventListener("click", () => {
    $("seed").value = Math.floor(Math.random() * 10000);
    loadMaze();
  });
  $("seed").addEventListener("change", loadMaze);
  $("play").addEventListener("click", () => setPlaying(!state.playing));
  $("scrub").addEventListener("input", (e) => {
    state.frame = parseInt(e.target.value, 10);
    setPlaying(false);
    $("clock").textContent = fmt(state.result?.trace[currentFrame()].t ?? 0) + " s";
    drawMaze();
  });
  for (const id of ["showOptimal", "showVisited"]) {
    $(id).addEventListener("change", drawMaze);
  }
  window.addEventListener("resize", drawMaze);
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", drawMaze);

  await loadMaze();
  requestAnimationFrame(tick);
}

init();
