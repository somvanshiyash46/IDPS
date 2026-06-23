#!/usr/bin/env python3
"""
THREATSHIELD — dashboard.py
Simple Flask web dashboard. Reads events.json written by idps_engine.py
and serves a live monitoring page on port 5000.

Run: python3 dashboard.py
"""

import json
import os
from flask import Flask, jsonify, render_template_string

app = Flask(__name__)

EVENTS_JSON = "/opt/threatshield/events.json"
BLOCKED_LOG = "/opt/threatshield/idps.log"

# ─────────────────────────────────────────────────────────────
# HTML DASHBOARD  (single-file, no external CSS/JS dependencies)
# ─────────────────────────────────────────────────────────────
DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta http-equiv="refresh" content="5">  <!-- auto-refresh every 5 sec -->
<title>THREATSHIELD — Live Dashboard</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: #0d1117; color: #c9d1d9; font-family: 'Courier New', monospace; font-size: 13px; }

  /* ── TOP BAR ── */
  .topbar {
    background: linear-gradient(135deg, #1b2a4a 0%, #0d1117 100%);
    border-bottom: 2px solid #1e3a5f;
    padding: 14px 28px;
    display: flex; align-items: center; justify-content: space-between;
  }
  .topbar .logo { font-size: 20px; font-weight: bold; color: #58a6ff; letter-spacing: 2px; }
  .topbar .subtitle { color: #7d8590; font-size: 11px; margin-top: 2px; }
  .live-dot { display: inline-block; width: 10px; height: 10px; background: #3fb950;
              border-radius: 50%; margin-right: 6px; animation: blink 1s infinite; }
  @keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.2} }
  .status-live { color: #3fb950; font-size: 12px; }

  /* ── STAT CARDS ── */
  .cards { display: flex; gap: 16px; padding: 20px 28px; flex-wrap: wrap; }
  .card {
    flex: 1; min-width: 160px; background: #161b22;
    border: 1px solid #30363d; border-radius: 10px; padding: 18px 22px;
    text-align: center;
  }
  .card .val { font-size: 36px; font-weight: bold; margin: 6px 0; }
  .card .lbl { font-size: 11px; color: #7d8590; text-transform: uppercase; letter-spacing: 1px; }
  .card.total  .val { color: #58a6ff; }
  .card.attack .val { color: #f85149; }
  .card.block  .val { color: #ff7b72; }
  .card.benign .val { color: #3fb950; }

  /* ── SECTION ── */
  .section { padding: 0 28px 20px 28px; }
  .section h2 { font-size: 13px; color: #7d8590; text-transform: uppercase;
                letter-spacing: 1px; margin-bottom: 10px; border-bottom: 1px solid #21262d; padding-bottom: 6px; }

  /* ── EVENT TABLE ── */
  table { width: 100%; border-collapse: collapse; }
  th { background: #161b22; color: #7d8590; font-size: 11px; text-transform: uppercase;
       letter-spacing: 0.5px; padding: 8px 12px; text-align: left; border-bottom: 1px solid #30363d; }
  td { padding: 7px 12px; border-bottom: 1px solid #21262d; font-size: 12px; }
  tr:hover td { background: #1c2128; }
  .badge { display: inline-block; padding: 2px 8px; border-radius: 12px; font-size: 11px; font-weight: bold; }
  .badge.attack { background: rgba(248,81,73,0.15); color: #f85149; border: 1px solid #f85149; }
  .badge.benign { background: rgba(63,185,80,0.15); color: #3fb950; border: 1px solid #3fb950; }
  .badge.block  { background: rgba(255,123,114,0.2); color: #ff7b72; border: 1px solid #ff7b72; }

  .prob-bar { background: #21262d; border-radius: 4px; height: 8px; overflow: hidden; width: 80px; display: inline-block; vertical-align: middle; margin-right: 6px; }
  .prob-fill { height: 100%; border-radius: 4px; }

  /* ── BLOCKED IP LIST ── */
  .blocked-grid { display: flex; flex-wrap: wrap; gap: 8px; }
  .ip-chip { background: rgba(248,81,73,0.1); border: 1px solid #f85149;
             color: #f85149; padding: 4px 12px; border-radius: 20px; font-size: 12px; }
  .ip-chip.none { color: #7d8590; border-color: #30363d; background: transparent; }

  /* ── FOOTER ── */
  .footer { text-align: center; padding: 14px; color: #4a4f54; font-size: 11px; }
</style>
</head>
<body>

<!-- TOP BAR -->
<div class="topbar">
  <div>
    <div class="logo">🛡 THREATSHIELD</div>
    <div class="subtitle">Industrial Network Intrusion Detection &amp; Prevention System</div>
  </div>
  <div>
    <span class="live-dot"></span>
    <span class="status-live">LIVE MONITORING</span>
    <div style="color:#4a4f54; font-size:11px; margin-top:3px;">Auto-refresh every 5s</div>
  </div>
</div>

<!-- STAT CARDS -->
<div class="cards">
  <div class="card total">
    <div class="val" id="total">{{ stats.total }}</div>
    <div class="lbl">Total Flows Analyzed</div>
  </div>
  <div class="card attack">
    <div class="val" id="attacks">{{ stats.attacks }}</div>
    <div class="lbl">Attacks Detected</div>
  </div>
  <div class="card block">
    <div class="val" id="blocked">{{ stats.blocked }}</div>
    <div class="lbl">IPs Blocked</div>
  </div>
  <div class="card benign">
    <div class="val" id="benign">{{ stats.benign }}</div>
    <div class="lbl">Benign Traffic</div>
  </div>
</div>

<!-- BLOCKED IP SECTION -->
<div class="section">
  <h2>🔴 Blocked IP Addresses ({{ blocked_ips | length }})</h2>
  <div class="blocked-grid">
    {% if blocked_ips %}
      {% for ip in blocked_ips %}
        <span class="ip-chip">{{ ip }}</span>
      {% endfor %}
    {% else %}
      <span class="ip-chip none">No IPs blocked yet</span>
    {% endif %}
  </div>
</div>

<!-- RECENT EVENTS TABLE -->
<div class="section">
  <h2>📋 Recent Detection Events (Last {{ events | length }})</h2>
  <table>
    <thead>
      <tr>
        <th>Timestamp (UTC)</th>
        <th>Source IP</th>
        <th>Attack Probability</th>
        <th>Label</th>
        <th>Strike</th>
        <th>Action</th>
      </tr>
    </thead>
    <tbody>
      {% for e in events %}
      <tr>
        <td style="color:#8b949e">{{ e.timestamp }}</td>
        <td style="color:#e6edf3; font-weight:bold">{{ e.src_ip }}</td>
        <td>
          <div class="prob-bar">
            <div class="prob-fill" style="width:{{ (e.prob * 100)|int }}%;
              background: {% if e.prob >= 0.4 %}#f85149{% else %}#3fb950{% endif %}">
            </div>
          </div>
          <span style="color:{% if e.prob >= 0.4 %}#f85149{% else %}#3fb950{% endif %}">
            {{ "%.1f"|format(e.prob * 100) }}%
          </span>
        </td>
        <td>
          <span class="badge {% if e.label == 'ATTACK' %}attack{% else %}benign{% endif %}">
            {{ e.label }}
          </span>
        </td>
        <td style="color:#e6edf3">{{ e.strike }}</td>
        <td>
          {% if 'BLOCKED' in e.action %}
            <span class="badge block">⛔ BLOCKED</span>
          {% elif e.label == 'ATTACK' %}
            <span style="color:#f0883e">⚠ {{ e.action }}</span>
          {% else %}
            <span style="color:#3fb950">✓ Allow</span>
          {% endif %}
        </td>
      </tr>
      {% endfor %}
      {% if not events %}
      <tr><td colspan="6" style="text-align:center; color:#4a4f54; padding:24px">
        Waiting for network traffic... (IDPS engine starting up)
      </td></tr>
      {% endif %}
    </tbody>
  </table>
</div>

<div class="footer">
  THREATSHIELD v1.0 &nbsp;|&nbsp; XGBoost Classifier (71 features, CICIoT 2025)
  &nbsp;|&nbsp; Threshold: 0.40 &nbsp;|&nbsp; 3-Strike Block Policy
</div>

</body>
</html>
"""

# ─────────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────────
def load_events():
    """Load events.json written by idps_engine.py"""
    try:
        with open(EVENTS_JSON) as f:
            events = json.load(f)
        return list(reversed(events[-50:]))  # newest first, last 50
    except Exception:
        return []

@app.route("/")
def dashboard():
    events = load_events()
    stats = events[0]["stats"] if events else {"total": 0, "attacks": 0, "blocked": 0, "benign": 0}
    blocked_ips = list({e["src_ip"] for e in events if e.get("action") and "BLOCKED" in e["action"]})
    return render_template_string(
        DASHBOARD_HTML,
        events=events,
        stats=stats,
        blocked_ips=blocked_ips,
    )

@app.route("/api/events")
def api_events():
    """JSON API — for programmatic access or future JS auto-refresh"""
    return jsonify(load_events())

@app.route("/api/stats")
def api_stats():
    events = load_events()
    stats = events[0]["stats"] if events else {"total": 0, "attacks": 0, "blocked": 0, "benign": 0}
    return jsonify(stats)

# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("🛡 THREATSHIELD Dashboard starting on http://0.0.0.0:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)
