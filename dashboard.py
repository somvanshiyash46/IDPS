#!/usr/bin/env python3
"""
THREATSHIELD — dashboard.py  v2.0
Flask web dashboard. Reads events.json written by idps_engine.py.
Serves a live monitoring page on port 5000.

Improvements over v1:
  - JS fetch() auto-refresh instead of <meta http-equiv="refresh"> (no full-page flicker)
  - Attack trend sparkline (last 12 × 5-second windows)
  - Top-5 attacking IPs table
  - Manual unblock button (calls POST /api/unblock/<ip>)
  - /api/health endpoint for load-balancer / uptime checks
  - Handles missing or malformed events.json gracefully
  - Static files served from memory (single-file deploy, no external CDN)
"""

import json
import os
import subprocess
from datetime import datetime, timezone
from collections import Counter, deque

from flask import Flask, jsonify, render_template_string, abort, request

app = Flask(__name__)

EVENTS_JSON = os.environ.get("TS_JSON_PATH", "/opt/threatshield/events.json")
BLOCKED_LOG = os.environ.get("TS_LOG_PATH",  "/opt/threatshield/idps.log")

# ─────────────────────────────────────────────────────────────
# HTML / CSS / JS  (single self-contained file)
# ─────────────────────────────────────────────────────────────
DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>THREATSHIELD — Live Dashboard</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{background:#0d1117;color:#c9d1d9;font-family:'Courier New',monospace;font-size:13px}

/* TOP BAR */
.topbar{background:#161b22;border-bottom:2px solid #1e3a5f;padding:14px 24px;display:flex;align-items:center;justify-content:space-between;gap:16px}
.logo{font-size:19px;font-weight:bold;color:#58a6ff;letter-spacing:2px}
.subtitle{color:#7d8590;font-size:11px;margin-top:2px}
.live-dot{display:inline-block;width:9px;height:9px;background:#3fb950;border-radius:50%;margin-right:5px;animation:blink 1.2s infinite}
@keyframes blink{0%,100%{opacity:1}50%{opacity:.2}}
.status-text{color:#3fb950;font-size:12px}
.last-refresh{color:#4a4f54;font-size:11px;margin-top:3px}

/* STAT CARDS */
.cards{display:flex;gap:14px;padding:18px 24px;flex-wrap:wrap}
.card{flex:1;min-width:140px;background:#161b22;border:1px solid #30363d;border-radius:10px;padding:16px 20px;text-align:center;transition:border-color .2s}
.card:hover{border-color:#58a6ff}
.card .val{font-size:34px;font-weight:bold;margin:5px 0;transition:color .3s}
.card .lbl{font-size:10px;color:#7d8590;text-transform:uppercase;letter-spacing:1px}
.card.total  .val{color:#58a6ff}
.card.attack .val{color:#f85149}
.card.block  .val{color:#ff7b72}
.card.benign .val{color:#3fb950}

/* SPARKLINE */
.spark-wrap{padding:0 24px 16px}
.spark-title{font-size:11px;color:#7d8590;text-transform:uppercase;letter-spacing:1px;margin-bottom:6px}
#sparkline{width:100%;height:48px;display:block}

/* SECTION */
.section{padding:0 24px 20px}
.section h2{font-size:12px;color:#7d8590;text-transform:uppercase;letter-spacing:1px;margin-bottom:10px;border-bottom:1px solid #21262d;padding-bottom:6px;display:flex;align-items:center;gap:8px}

/* SEARCH */
.search-bar{margin-bottom:10px}
.search-bar input{background:#161b22;border:1px solid #30363d;color:#c9d1d9;padding:6px 12px;border-radius:6px;font-size:12px;font-family:inherit;width:260px;outline:none}
.search-bar input:focus{border-color:#58a6ff}

/* TABLE */
table{width:100%;border-collapse:collapse}
th{background:#161b22;color:#7d8590;font-size:10px;text-transform:uppercase;letter-spacing:.5px;padding:8px 10px;text-align:left;border-bottom:1px solid #30363d;position:sticky;top:0}
td{padding:7px 10px;border-bottom:1px solid #21262d;font-size:12px}
tr:hover td{background:#1c2128}
.badge{display:inline-block;padding:2px 7px;border-radius:10px;font-size:10px;font-weight:bold}
.badge.attack{background:rgba(248,81,73,.15);color:#f85149;border:1px solid #f85149}
.badge.benign{background:rgba(63,185,80,.15);color:#3fb950;border:1px solid #3fb950}
.badge.block {background:rgba(255,123,114,.2);color:#ff7b72;border:1px solid #ff7b72}
.prob-bar{background:#21262d;border-radius:3px;height:7px;overflow:hidden;width:70px;display:inline-block;vertical-align:middle;margin-right:5px}
.prob-fill{height:100%;border-radius:3px}

/* BLOCKED IPs */
.blocked-grid{display:flex;flex-wrap:wrap;gap:8px}
.ip-chip{background:rgba(248,81,73,.1);border:1px solid #f85149;color:#f85149;padding:4px 10px;border-radius:16px;font-size:12px;display:flex;align-items:center;gap:6px}
.ip-chip button{background:none;border:none;color:#f85149;cursor:pointer;font-size:14px;line-height:1;padding:0;opacity:.6;transition:opacity .15s}
.ip-chip button:hover{opacity:1}
.ip-chip.none{color:#7d8590;border-color:#30363d;background:transparent}

/* TOP ATTACKERS */
.top-tbl td:first-child{color:#f85149;font-weight:bold}
.top-tbl td:last-child{color:#7d8590}

/* TWO-COL LAYOUT */
.row2{display:flex;gap:20px;padding:0 24px 20px}
.row2 .col{flex:1;min-width:0}

/* FOOTER */
.footer{text-align:center;padding:14px;color:#4a4f54;font-size:11px}
</style>
</head>
<body>

<div class="topbar">
  <div>
    <div class="logo">🛡 THREATSHIELD</div>
    <div class="subtitle">Industrial Network Intrusion Detection &amp; Prevention System</div>
  </div>
  <div style="text-align:right">
    <div><span class="live-dot"></span><span class="status-text">LIVE MONITORING</span></div>
    <div class="last-refresh" id="last-refresh">Connecting...</div>
  </div>
</div>

<!-- STAT CARDS -->
<div class="cards">
  <div class="card total" ><div class="val" id="c-total" >—</div><div class="lbl">Total Flows</div></div>
  <div class="card attack"><div class="val" id="c-attack">—</div><div class="lbl">Attacks Detected</div></div>
  <div class="card block" ><div class="val" id="c-block" >—</div><div class="lbl">IPs Blocked</div></div>
  <div class="card benign"><div class="val" id="c-benign">—</div><div class="lbl">Benign Traffic</div></div>
</div>

<!-- SPARKLINE -->
<div class="spark-wrap">
  <div class="spark-title">Attack detections — last 60 seconds (12 × 5 s)</div>
  <canvas id="sparkline"></canvas>
</div>

<!-- BLOCKED IPs -->
<div class="section">
  <h2>🔴 Blocked IP addresses <span id="blocked-count" style="color:#c9d1d9"></span></h2>
  <div class="blocked-grid" id="blocked-grid">
    <span class="ip-chip none">Loading...</span>
  </div>
</div>

<!-- TWO COLUMNS: events table + top attackers -->
<div class="row2">
  <div class="col" style="flex:2">
    <div class="section" style="padding-left:0;padding-right:0">
      <h2>📋 Recent detection events <span id="event-count" style="color:#c9d1d9"></span></h2>
      <div class="search-bar">
        <input type="text" id="search" placeholder="Filter by IP or action..." oninput="renderTable()">
      </div>
      <div style="overflow-x:auto">
        <table id="events-tbl">
          <thead>
            <tr>
              <th>Timestamp (UTC)</th>
              <th>Source IP</th>
              <th>Probability</th>
              <th>Label</th>
              <th>Strike</th>
              <th>Action</th>
            </tr>
          </thead>
          <tbody id="events-body"></tbody>
        </table>
      </div>
    </div>
  </div>

  <div class="col" style="flex:1">
    <div class="section" style="padding-left:0;padding-right:0">
      <h2>🏴 Top attacking IPs</h2>
      <table class="top-tbl">
        <thead><tr><th>IP</th><th>Attacks</th><th>Status</th></tr></thead>
        <tbody id="top-body"></tbody>
      </table>
    </div>
  </div>
</div>

<div class="footer">
  THREATSHIELD v2.0 &nbsp;|&nbsp; XGBoost (71 features, CICIoT 2025)
  &nbsp;|&nbsp; Threshold: {{ threshold }} &nbsp;|&nbsp; Strike limit: {{ strike_limit }}
</div>

<script>
let _events = [];
let _sparkData = Array(12).fill(0);
const REFRESH_MS = 5000;

async function refresh() {
  try {
    const r = await fetch('/api/events');
    if (!r.ok) throw new Error(r.status);
    _events = await r.json();
    updateCards();
    updateSparkline();
    updateBlocked();
    updateTopAttackers();
    renderTable();
    document.getElementById('last-refresh').textContent =
      'Last update: ' + new Date().toLocaleTimeString();
  } catch(e) {
    document.getElementById('last-refresh').textContent = 'Refresh error — retrying...';
  }
}

function updateCards() {
  const s = _events.length ? _events[0].stats : {total:0,attacks:0,blocked:0,benign:0};
  document.getElementById('c-total' ).textContent = s.total;
  document.getElementById('c-attack').textContent = s.attacks;
  document.getElementById('c-block' ).textContent = s.blocked;
  document.getElementById('c-benign').textContent = s.benign;
}

function updateSparkline() {
  // Count attacks in last 12 windows (most recent window = last entry timestamp bucket)
  const now = Date.now() / 1000;
  const buckets = Array(12).fill(0);
  _events.forEach(e => {
    if (e.label !== 'ATTACK') return;
    const ts = new Date(e.timestamp + 'Z').getTime() / 1000;
    const age = now - ts;
    const idx = Math.floor(age / 5);
    if (idx >= 0 && idx < 12) buckets[11 - idx]++;
  });
  _sparkData = buckets;
  drawSparkline();
}

function drawSparkline() {
  const c = document.getElementById('sparkline');
  const W = c.offsetWidth || 600, H = 48;
  c.width = W; c.height = H;
  const ctx = c.getContext('2d');
  ctx.clearRect(0,0,W,H);
  const max = Math.max(..._sparkData, 1);
  const bw  = W / 12;
  _sparkData.forEach((v, i) => {
    const bh = Math.max(2, (v / max) * (H - 6));
    ctx.fillStyle = v > 0 ? 'rgba(248,81,73,0.7)' : 'rgba(63,185,80,0.25)';
    ctx.beginPath();
    ctx.roundRect(i * bw + 2, H - bh, bw - 4, bh, 2);
    ctx.fill();
  });
}

function updateBlocked() {
  const ips = [...new Set(_events.filter(e => e.action && e.action.includes('BLOCKED')).map(e => e.src_ip))];
  const grid = document.getElementById('blocked-grid');
  document.getElementById('blocked-count').textContent = `(${ips.length})`;
  if (!ips.length) {
    grid.innerHTML = '<span class="ip-chip none">No IPs blocked yet</span>';
    return;
  }
  grid.innerHTML = ips.map(ip =>
    `<span class="ip-chip">${ip}
       <button onclick="unblockIP('${ip}')" title="Unblock">✕</button>
     </span>`
  ).join('');
}

function updateTopAttackers() {
  const counts = {};
  _events.filter(e => e.label === 'ATTACK').forEach(e => {
    counts[e.src_ip] = (counts[e.src_ip] || 0) + 1;
  });
  const top5 = Object.entries(counts).sort((a,b) => b[1]-a[1]).slice(0,5);
  const blockedSet = new Set(
    _events.filter(e => e.action && e.action.includes('BLOCKED')).map(e => e.src_ip)
  );
  const tbody = document.getElementById('top-body');
  if (!top5.length) {
    tbody.innerHTML = '<tr><td colspan="3" style="color:#4a4f54;padding:12px">No attacks yet</td></tr>';
    return;
  }
  tbody.innerHTML = top5.map(([ip, n]) =>
    `<tr>
       <td>${ip}</td>
       <td>${n}</td>
       <td>${blockedSet.has(ip)
             ? '<span class="badge block">BLOCKED</span>'
             : '<span style="color:#7d8590">active</span>'}</td>
     </tr>`
  ).join('');
}

function renderTable() {
  const q = (document.getElementById('search').value || '').toLowerCase();
  const rows = _events.filter(e =>
    !q || e.src_ip.includes(q) || (e.action || '').toLowerCase().includes(q)
  ).slice(0, 50);

  document.getElementById('event-count').textContent = `(${rows.length})`;

  if (!rows.length) {
    document.getElementById('events-body').innerHTML =
      '<tr><td colspan="6" style="text-align:center;color:#4a4f54;padding:24px">No events yet</td></tr>';
    return;
  }

  document.getElementById('events-body').innerHTML = rows.map(e => {
    const pct  = Math.round(e.prob * 100);
    const col  = e.prob >= 0.4 ? '#f85149' : '#3fb950';
    const labelBadge = e.label === 'ATTACK'
      ? '<span class="badge attack">ATTACK</span>'
      : '<span class="badge benign">BENIGN</span>';
    let actionHtml;
    if (e.action && e.action.includes('BLOCKED'))
      actionHtml = '<span class="badge block">⛔ BLOCKED</span>';
    else if (e.label === 'ATTACK')
      actionHtml = `<span style="color:#f0883e">⚠ ${e.action}</span>`;
    else
      actionHtml = '<span style="color:#3fb950">✓ Allow</span>';

    return `<tr>
      <td style="color:#8b949e">${e.timestamp}</td>
      <td style="color:#e6edf3;font-weight:bold">${e.src_ip}</td>
      <td>
        <div class="prob-bar"><div class="prob-fill" style="width:${pct}%;background:${col}"></div></div>
        <span style="color:${col}">${pct}%</span>
      </td>
      <td>${labelBadge}</td>
      <td style="color:#e6edf3">${e.strike}</td>
      <td>${actionHtml}</td>
    </tr>`;
  }).join('');
}

async function unblockIP(ip) {
  if (!confirm(`Unblock ${ip}?`)) return;
  try {
    const r = await fetch(`/api/unblock/${encodeURIComponent(ip)}`, {method:'POST'});
    const j = await r.json();
    alert(j.message || JSON.stringify(j));
    await refresh();
  } catch(e) {
    alert('Unblock request failed: ' + e);
  }
}

window.addEventListener('resize', drawSparkline);
refresh();
setInterval(refresh, REFRESH_MS);
</script>
</body>
</html>
"""

# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────
def load_events() -> list:
    try:
        with open(EVENTS_JSON) as f:
            events = json.load(f)
        if not isinstance(events, list):
            return []
        return list(reversed(events[-100:]))  # newest first, last 100
    except (FileNotFoundError, json.JSONDecodeError):
        return []
    except Exception:
        return []


def _read_config() -> dict:
    """Pull runtime config from the first event's embedded stats, with fallbacks."""
    events = load_events()
    return {
        "threshold":    os.environ.get("TS_THRESHOLD",     "0.75"),
        "strike_limit": os.environ.get("TS_STRIKE_LIMIT",  "2"),
    }

# ─────────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────────
@app.route("/")
def dashboard():
    cfg = _read_config()
    return render_template_string(DASHBOARD_HTML, **cfg)


@app.route("/api/events")
def api_events():
    return jsonify(load_events())


@app.route("/api/stats")
def api_stats():
    events = load_events()
    stats = events[0]["stats"] if events else {"total": 0, "attacks": 0, "blocked": 0, "benign": 0}
    return jsonify(stats)


@app.route("/api/health")
def api_health():
    """Simple health-check endpoint for load balancers / uptime monitors."""
    events = load_events()
    return jsonify({
        "status": "ok",
        "events_loaded": len(events),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })


@app.route("/api/unblock/<ip>", methods=["POST"])
def api_unblock(ip: str):
    """
    Manually remove an iptables block for the given IP.
    The engine's in-memory blocked_ips set is NOT updated here (it lives in a
    separate process), so the engine may re-block on the next attack cycle.
    For a permanent unblock, restart the IDPS engine after removing the rule.
    """
    try:
        result = subprocess.run(
            ["iptables", "-D", "INPUT", "-s", ip, "-j", "DROP"],
            capture_output=True, timeout=5,
        )
        if result.returncode == 0:
            return jsonify({"status": "ok", "message": f"iptables rule removed for {ip}"})
        else:
            return jsonify({
                "status": "warning",
                "message": f"iptables returned {result.returncode} — rule may not have existed",
            })
    except subprocess.TimeoutExpired:
        return jsonify({"status": "error", "message": "iptables timed out"}), 500
    except Exception as exc:
        return jsonify({"status": "error", "message": str(exc)}), 500


# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("🛡 THREATSHIELD Dashboard v2.0 starting on http://0.0.0.0:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)
