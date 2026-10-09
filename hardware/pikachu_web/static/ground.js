"use strict";
// P2c-0 Ground Control 前端。人工调试/监视；非最终控制链。
const $ = (id) => document.getElementById(id);
const SPEED = () => Number.parseFloat($("speed").value);
const DIRS = { forward: [1, 0], backward: [-1, 0], left: [0, 1], right: [0, -1] };
const WHEEL_NAMES = ["L", "FL", "F", "FR", "R"];

let mode = "AUTO", estop = false, streamsOk = false, serialOpen = false;
let activeDir = null, holdTimer = null, pulseTimer = null, hbTimer = null, pollTimer = null;

async function api(path, body) {
  const opt = body === undefined
    ? { method: "GET", cache: "no-store" }
    : { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) };
  const res = await fetch(path, opt);
  let data = {};
  try { data = await res.json(); } catch (e) { /* ignore */ }
  return { httpOk: res.ok, status: res.status, data };
}

function applyAuth(a) {
  if (!a) return;
  if (a.mode) mode = a.mode;
  if (typeof a.estop === "boolean") estop = a.estop;
  const manual = mode === "MANUAL" && !estop;
  const usable = manual && streamsOk;

  const pill = $("modePill");
  pill.textContent = mode;
  pill.className = "pill " + (mode === "MANUAL" ? "manual" : "auto");
  const es = $("estopState");
  es.textContent = "E-STOP: " + (estop ? "ON" : "OFF");
  es.classList.toggle("on", estop);
  $("clearEstopBtn").disabled = !estop;

  document.querySelectorAll(".gc-dir[data-dir]").forEach((b) => { b.disabled = !usable; });

  const banner = $("banner");
  if (estop) {
    banner.textContent = "E-STOP 已触发并 latch：所有运动被挡死，点“清除 E-STOP”后再继续。";
    banner.className = "gc-banner estop";
  } else if (mode === "MANUAL" && !streamsOk) {
    banner.textContent = "MANUAL 已接管，但 odom/ToF stale 或 unhealthy —— 手动运动被禁止。";
    banner.className = "gc-banner manual";
  } else if (mode === "MANUAL") {
    banner.textContent = "MANUAL：网页方向键可用；/api/drive（Windows 链路）已被联锁挡住。";
    banner.className = "gc-banner manual";
  } else {
    banner.textContent = "只读监视中（AUTO）。点“接管 MANUAL”后才能手动驱动。";
    banner.className = "gc-banner";
  }
}

function renderConn(s) {
  serialOpen = !!(s.serial && s.serial.open);
  $("connDot").className = "status-dot " + (serialOpen ? "ok" : "bad");
  $("connText").textContent = serialOpen ? "串口已连接" : "串口未连接";
  const sr = s.serial || {};
  $("connMeta").textContent = sr.port ? `${sr.port} @ ${sr.baud}` : (sr.last_error || "no device");
}

function renderLink(s) {
  const rows = [];
  const od = s.odom || {}, tf = s.tof || {};
  const ageState = (ok, age) =>
    ok ? `<span class="gc-ok">OK</span>` : `<span class="gc-bad">STALE</span>` +
         (age != null ? ` <span class="gc-meta">(${age}s)</span>` : "");
  rows.push(["odom", ageState(od.ok, od.age), od.session_id || "-", `seq ${od.seq ?? "-"}`]);
  rows.push(["tof", ageState(tf.ok, tf.age), tf.session_id || "-", tf.healthy ? "healthy" : "unhealthy"]);
  rows.push(["serial", serialOpen ? '<span class="gc-ok">OPEN</span>' : '<span class="gc-bad">CLOSED</span>',
             (s.serial && s.serial.port) || "-", (s.serial && s.serial.last_error) || ""]);
  $("linkBody").innerHTML = rows.map((r) =>
    `<tr><td>${r[0]}</td><td class="num">${r[1]}</td><td>${r[2]}</td><td class="gc-meta">${r[3]}</td></tr>`
  ).join("");
}

function renderTof(tf) {
  const ranges = tf.ranges || {}, status = tf.status || {}, ages = tf.ages_ms || {};
  $("tofBody").innerHTML = WHEEL_NAMES.map((n) => {
    const mm = ranges[n];
    const st = status[n] || "-";
    const cls = st === "VALID" ? "gc-ok" : (st === "IO_ERROR" ? "gc-bad" : "gc-meta");
    return `<tr><td>${n}</td><td class="num">${mm == null ? "----" : mm}</td>` +
           `<td class="${cls}">${st}</td><td class="num">${ages[n] == null ? "-" : ages[n]}</td></tr>`;
  }).join("");
  $("tofMeta").textContent =
    `age ${tf.age == null ? "-" : tf.age + "s"} | healthy=${tf.healthy} | ` +
    `init=${tf.init_count ?? "-"} rerr=${tf.read_errors ?? "-"} reinit=${tf.reinit_count ?? "-"}`;
}

function renderOdom(od) {
  const rows = [
    ["x (m)", od.x], ["y (m)", od.y], ["theta (deg)", od.theta == null ? "-" : (od.theta * 180 / Math.PI).toFixed(1)],
    ["left ticks", od.left], ["right ticks", od.right],
    ["age", od.age == null ? "-" : od.age + "s"],
  ];
  $("odomBody").innerHTML = rows.map((r) =>
    `<tr><td>${r[0]}</td><td class="num">${r[1] == null ? "-" : r[1]}</td></tr>`).join("");
}

function renderWall(w) {
  if (!w) return;
  $("wallState").textContent = w.running
    ? `running… (stop_mm=${w.stop_mm})`
    : (w.reason ? `stopped: ${w.reason}` : "idle");
}

async function poll() {
  try {
    const r = await api("/api/manual/state");
    const s = r.data || {};
    streamsOk = !!s.streams_ok;
    renderConn(s); renderLink(s);
    renderTof(s.tof || {}); renderOdom(s.odom || {}); renderWall(s.wall_run);
    applyAuth(s.authority);
  } catch (e) {
    $("connText").textContent = "服务器离线";
    $("connDot").className = "status-dot bad";
  }
}

// ---- 命令 ----
function drive(v, w) {
  return api("/api/manual/drive", { v, w });
}
function stopMotion() {
  stopHold();
  return api("/api/manual/stop", {});
}
function startHeartbeat() {
  if (hbTimer) return;
  hbTimer = setInterval(() => api("/api/manual/heartbeat", {}), 2000);
}
function stopHeartbeat() { if (hbTimer) { clearInterval(hbTimer); hbTimer = null; } }

async function takeControl() {
  const r = await api("/api/manual/mode", { mode: "MANUAL" });
  applyAuth(r.data);
  startHeartbeat();
}
async function releaseControl() {
  stopHold();
  await api("/api/manual/release", {});
  stopHeartbeat();
  poll();
}

function pulseDuration() {
  const el = document.querySelector('input[name="pdur"]:checked');
  return el ? Number.parseFloat(el.value) : 0.5;
}

function stopHold() {
  if (holdTimer) { clearInterval(holdTimer); holdTimer = null; }
  if (pulseTimer) { clearTimeout(pulseTimer); pulseTimer = null; }
  activeDir = null;
  document.querySelectorAll(".gc-dir.active").forEach((b) => b.classList.remove("active"));
}

function pressDir(btn, dir) {
  if (btn.disabled) return;
  const s = SPEED();
  const [v, w] = DIRS[dir];
  btn.classList.add("active");
  activeDir = dir;
  if ($("pulseMode").checked) {
    // 定时脉冲 —— 自动 STOP
    drive(v * s, w * s);
    pulseTimer = setTimeout(() => { pulseTimer = null; stopMotion(); btn.classList.remove("active"); activeDir = null; },
                            pulseDuration() * 1000);
    return;
  }
  drive(v * s, w * s);
  holdTimer = setInterval(() => drive(v * s, w * s), 180);
}

// ---- 绑定 ----
function bind() {
  $("speed").addEventListener("input", () => { $("speedVal").textContent = SPEED().toFixed(2); });

  document.querySelectorAll(".gc-dir[data-dir]").forEach((btn) => {
    const dir = btn.dataset.dir;
    btn.addEventListener("pointerdown", (e) => { e.preventDefault(); pressDir(btn, dir); });
    ["pointerup", "pointerleave", "pointercancel"].forEach((ev) =>
      btn.addEventListener(ev, (e) => { e.preventDefault(); stopMotion(); }));
  });
  $("stopBtn").addEventListener("click", () => stopMotion());

  $("takeBtn").addEventListener("click", takeControl);
  $("releaseBtn").addEventListener("click", releaseControl);
  $("estopBtn").addEventListener("click", async () => {
    stopHold();
    const r = await api("/api/manual/estop", { reason: "web" });
    applyAuth(r.data);
  });
  $("clearEstopBtn").addEventListener("click", async () => {
    const r = await api("/api/manual/clear-estop", {});
    applyAuth(r.data);
  });

  $("wallStart").addEventListener("click", async () => {
    const mag = Number.parseFloat($("wallMag").value);
    const stop_mm = Number.parseFloat($("wallStop").value);
    const max_s = Number.parseFloat($("wallMax").value);
    const r = await api("/api/manual/wall", { mag, stop_mm, max_s });
    if (!r.httpOk) { $("wallState").textContent = "拒绝: " + (r.data.error || r.status); }
  });
  $("wallAbort").addEventListener("click", () => api("/api/manual/wall-stop", {}));

  // 安全：失焦 / 切后台 / 关闭 -> STOP + 释放本页 MANUAL ownership（mode→AUTO）。
  // 只释放本页的占有；绝不去动 Windows bridge 自己的 FAILSAFE latch（那是另一侧的事）。
  const stopAndRelease = () => {
    stopHold();
    stopHeartbeat();
    if (navigator.sendBeacon) navigator.sendBeacon("/api/manual/release");
    else api("/api/manual/release", {});
  };
  window.addEventListener("blur", stopAndRelease);
  document.addEventListener("visibilitychange", () => { if (document.hidden) stopAndRelease(); });
  window.addEventListener("pagehide", stopAndRelease);
  window.addEventListener("beforeunload", stopAndRelease);
}

bind();
poll();
pollTimer = setInterval(poll, 200);
