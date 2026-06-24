#!/usr/bin/env python3
"""
THREATSHIELD — idps_engine.py  v2.0
Real-time packet capture, flow aggregation, XGBoost inference, and IP blocking.
Runs as a systemd service on AWS EC2. Writes detections to events.json for the dashboard.

Improvements over v1:
  - Thread-safe stats using threading.Lock (no more silent race-condition counter drift)
  - Graceful SIGTERM/SIGINT shutdown: flushes in-flight flows before exit
  - IP auto-unblock after BLOCK_DURATION seconds (configurable, default 10 min)
  - Strike decay: idle IPs shed a strike every STRIKE_DECAY_INTERVAL seconds
  - Atomic JSON write via temp-file rename (no half-written events.json on crash)
  - Interface auto-detection fallback: tries eth0, ens5, enp0s3 in order
  - Port 5001 (custom sensor simulator) added to allowlist
  - Fixed: mss_src / mss_dst variables were aliases of src_port / dst_port — removed
  - Fixed: network_protocols_dst_count and _src_count were identical (both used
    set(protocols)) — now correctly split by direction
  - Fixed: aggregate_flow returned early with bare zeros when packets=[] but FEATURES
    may not be populated yet at import time — guard added
  - Configurable via environment variables (no need to edit source)
"""

import os
import sys
import time
import json
import signal
import logging
import tempfile
import threading
import subprocess
import collections
from datetime import datetime, timezone

import joblib
import pandas as pd
import numpy as np
from scapy.all import sniff, IP, TCP, UDP, Ether, conf, get_if_list

# ─────────────────────────────────────────────────────────────
# CONFIG  (all overridable via environment variables)
# ─────────────────────────────────────────────────────────────
MODEL_PATH      = os.environ.get("TS_MODEL_PATH",    "/opt/threatshield/idps_xgboost_model.pkl")
LOG_PATH        = os.environ.get("TS_LOG_PATH",      "/opt/threatshield/idps.log")
JSON_LOG_PATH   = os.environ.get("TS_JSON_PATH",     "/opt/threatshield/events.json")
IFACE           = os.environ.get("TS_IFACE",         "")          # empty = auto-detect
THRESHOLD       = float(os.environ.get("TS_THRESHOLD",     "0.75"))
STRIKE_LIMIT    = int(os.environ.get("TS_STRIKE_LIMIT",    "2"))
FLOW_WINDOW     = int(os.environ.get("TS_FLOW_WINDOW",     "5"))
MAX_LOG_LINES   = int(os.environ.get("TS_MAX_LOG",         "500"))
MIN_FLOW_PKTS   = int(os.environ.get("TS_MIN_FLOW_PKTS",   "3"))
BLOCK_DURATION  = int(os.environ.get("TS_BLOCK_DURATION",  "600"))  # seconds; 0 = permanent
STRIKE_DECAY_IV = int(os.environ.get("TS_STRIKE_DECAY",    "60"))   # shed 1 strike/minute for quiet IPs

# This instance's own IPs — set by setup.sh via THREATSHIELD_LOCAL_IPS env var.
LOCAL_IPS = set(filter(None, os.environ.get("THREATSHIELD_LOCAL_IPS", "").split(",")))

# ─────────────────────────────────────────────────────────────
# ALLOWLISTS
# ─────────────────────────────────────────────────────────────
ALLOWLIST_IPS = {
    "169.254.169.254",   # AWS instance metadata service
    "169.254.169.253",   # AWS NTP / DNS resolver
}
ALLOWLIST_PORTS = {
    53,    # DNS
    123,   # NTP
    5001,  # custom sensor simulator (telemetry)
}

# ─────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH, mode="a"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("THREATSHIELD")

# ─────────────────────────────────────────────────────────────
# GLOBAL STATE  (all mutations guarded by _lock)
# ─────────────────────────────────────────────────────────────
_lock         = threading.Lock()
model         = None
FEATURES      = []
ip_strikes    = collections.defaultdict(int)
blocked_ips   = {}        # ip → unblock_timestamp (float); 0 = permanent
event_buffer  = collections.deque(maxlen=MAX_LOG_LINES)
flows         = collections.defaultdict(list)
stats         = {"total": 0, "attacks": 0, "blocked": 0, "benign": 0}
_shutdown     = threading.Event()

# ─────────────────────────────────────────────────────────────
# INTERFACE AUTO-DETECTION
# ─────────────────────────────────────────────────────────────
def detect_interface() -> str:
    """Return the first available non-loopback interface from a priority list."""
    if IFACE:
        return IFACE
    candidates = ["eth0", "ens5", "ens3", "enp0s3", "enp0s5"]
    available  = set(get_if_list())
    for iface in candidates:
        if iface in available:
            log.info(f"Auto-selected interface: {iface}")
            return iface
    # Last resort: first non-lo interface
    for iface in available:
        if iface != "lo":
            log.warning(f"Falling back to interface: {iface}")
            return iface
    raise RuntimeError("No usable network interface found")

# ─────────────────────────────────────────────────────────────
# MODEL LOAD
# ─────────────────────────────────────────────────────────────
def load_model():
    global model, FEATURES
    log.info(f"Loading model from {MODEL_PATH} ...")
    model    = joblib.load(MODEL_PATH)
    FEATURES = list(model.feature_names_in_)
    log.info(f"✅ Model loaded | features={len(FEATURES)} | classes={model.classes_}")

# ─────────────────────────────────────────────────────────────
# FLOW AGGREGATOR  →  71-feature dict
# ─────────────────────────────────────────────────────────────
def aggregate_flow(packets: list) -> dict:
    """Compute the 71 CICIoT-style statistical features for one flow window."""
    if not packets or not FEATURES:
        return {f: 0.0 for f in FEATURES} if FEATURES else {}

    def _vals(key):
        return [p[key] for p in packets if p.get(key) is not None]

    def safe_stat(vals):
        if not vals:
            return 0.0, 0.0, 0.0, 0.0
        arr = np.array(vals, dtype=float)
        return float(arr.mean()), float(arr.max()), float(arr.min()), float(arr.std())

    ttl_vals     = _vals("ttl")
    win_vals     = _vals("window")
    ip_len_vals  = _vals("ip_len")
    ip_flags_vals= _vals("ip_flags")
    pkt_size_vals= _vals("pkt_size")
    hdr_len_vals = _vals("hdr_len")
    pay_len_vals = _vals("pay_len")
    mss_vals     = _vals("mss")
    tcp_flags_v  = _vals("tcp_flags")
    timestamps   = [p["timestamp"] for p in packets]
    src_ips      = [p["src"] for p in packets]
    dst_ips      = [p["dst"] for p in packets]
    src_macs     = _vals("src_mac")
    dst_macs     = _vals("dst_mac")
    src_ports    = _vals("src_port")
    dst_ports    = _vals("dst_port")
    protocols    = _vals("protocol")

    # Time deltas
    time_deltas = [timestamps[i] - timestamps[i-1] for i in range(1, len(timestamps))]
    td_avg, td_max, td_min, td_std = safe_stat(time_deltas)

    # TCP flags
    syn_count = sum(1 for p in packets if p.get("flags_syn"))
    ack_count = sum(1 for p in packets if p.get("flags_ack"))
    fin_count = sum(1 for p in packets if p.get("flags_fin"))
    rst_count = sum(1 for p in packets if p.get("flags_rst"))
    psh_count = sum(1 for p in packets if p.get("flags_psh"))
    urg_count = sum(1 for p in packets if p.get("flags_urg"))
    tcf_avg, tcf_max, tcf_min, tcf_std = safe_stat(tcp_flags_v)

    # Fragmentation
    frag_pkts  = sum(1 for p in packets if p.get("is_fragment"))
    frag_score = frag_pkts / max(len(packets), 1)

    pay_avg, pay_max, pay_min, pay_std = safe_stat(pay_len_vals)
    hdr_avg, hdr_max, hdr_min, hdr_std = safe_stat(hdr_len_vals)
    ps_avg,  ps_max,  ps_min,  ps_std  = safe_stat(pkt_size_vals)
    il_avg,  il_max,  il_min,  il_std  = safe_stat(ip_len_vals)
    ipf_avg, ipf_max, ipf_min, ipf_std = safe_stat(ip_flags_vals)
    ttl_avg, ttl_max, ttl_min, ttl_std = safe_stat(ttl_vals)
    win_avg, win_max, win_min, win_std  = safe_stat(win_vals)
    mss_avg, mss_max, mss_min, mss_std  = safe_stat(mss_vals)

    n_pkts    = len(packets)
    # Direction-aware packet counts (was hardcoded outbound=all in v1)
    src_pkt_c = sum(1 for p in packets if p.get("direction") == "outbound")
    dst_pkt_c = sum(1 for p in packets if p.get("direction") == "inbound")
    if src_pkt_c == 0 and dst_pkt_c == 0:
        src_pkt_c = n_pkts

    # Protocol split by direction (was both identical set(protocols) in v1)
    proto_src = set(p.get("protocol") for p in packets if p.get("direction") == "outbound")
    proto_dst = set(p.get("protocol") for p in packets if p.get("direction") == "inbound")

    ldr_avg, ldr_max, ldr_min, ldr_std = safe_stat(pay_len_vals if pay_len_vals else [0.0])

    feat = {
        "log_data-ranges_avg":                  ldr_avg,
        "log_data-ranges_max":                  ldr_max,
        "log_data-ranges_min":                  ldr_min,
        "log_data-ranges_std_deviation":        ldr_std,
        "log_data-types_count":                 float(len(set(protocols))),
        "log_interval-messages":                td_avg,
        "log_messages_count":                   float(n_pkts),
        "network_fragmentation-score":          frag_score,
        "network_fragmented-packets":           float(frag_pkts),
        "network_header-length_avg":            hdr_avg,
        "network_header-length_max":            hdr_max,
        "network_header-length_min":            hdr_min,
        "network_header-length_std_deviation":  hdr_std,
        "network_interval-packets":             td_avg,
        "network_ip-flags_avg":                 ipf_avg,
        "network_ip-flags_max":                 ipf_max,
        "network_ip-flags_min":                 ipf_min,
        "network_ip-flags_std_deviation":       ipf_std,
        "network_ip-length_avg":                il_avg,
        "network_ip-length_max":                il_max,
        "network_ip-length_min":                il_min,
        "network_ip-length_std_deviation":      il_std,
        "network_ips_all_count":                float(len(set(src_ips + dst_ips))),
        "network_ips_dst_count":                float(len(set(dst_ips))),
        "network_ips_src_count":                float(len(set(src_ips))),
        "network_macs_all_count":               float(len(set(src_macs + dst_macs))),
        "network_macs_dst_count":               float(len(set(dst_macs))),
        "network_macs_src_count":               float(len(set(src_macs))),
        "network_mss_avg":                      mss_avg,
        "network_mss_max":                      mss_max,
        "network_mss_min":                      mss_min,
        "network_mss_std_deviation":            mss_std,
        "network_packet-size_avg":              ps_avg,
        "network_packet-size_max":              ps_max,
        "network_packet-size_min":              ps_min,
        "network_packet-size_std_deviation":    ps_std,
        "network_packets_all_count":            float(n_pkts),
        "network_packets_dst_count":            float(dst_pkt_c),
        "network_packets_src_count":            float(src_pkt_c),
        "network_payload-length_avg":           pay_avg,
        "network_payload-length_max":           pay_max,
        "network_payload-length_min":           pay_min,
        "network_payload-length_std_deviation": pay_std,
        "network_ports_all_count":              float(len(set(src_ports + dst_ports))),
        "network_ports_dst_count":              float(len(set(dst_ports))),
        "network_ports_src_count":              float(len(set(src_ports))),
        "network_protocols_all_count":          float(len(set(protocols))),
        "network_protocols_dst_count":          float(len(proto_dst)),   # fixed v1 bug
        "network_protocols_src_count":          float(len(proto_src)),   # fixed v1 bug
        "network_tcp-flags-ack_count":          float(ack_count),
        "network_tcp-flags-fin_count":          float(fin_count),
        "network_tcp-flags-psh_count":          float(psh_count),
        "network_tcp-flags-rst_count":          float(rst_count),
        "network_tcp-flags-syn_count":          float(syn_count),
        "network_tcp-flags-urg_count":          float(urg_count),
        "network_tcp-flags_avg":                tcf_avg,
        "network_tcp-flags_max":                tcf_max,
        "network_tcp-flags_min":                tcf_min,
        "network_tcp-flags_std_deviation":      tcf_std,
        "network_time-delta_avg":               td_avg,
        "network_time-delta_max":               td_max,
        "network_time-delta_min":               td_min,
        "network_time-delta_std_deviation":     td_std,
        "network_ttl_avg":                      ttl_avg,
        "network_ttl_max":                      ttl_max,
        "network_ttl_min":                      ttl_min,
        "network_ttl_std_deviation":            ttl_std,
        "network_window-size_avg":              win_avg,
        "network_window-size_max":              win_max,
        "network_window-size_min":              win_min,
        "network_window-size_std_deviation":    win_std,
    }

    # Safety net: fill any model feature absent from the computed dict
    for f in FEATURES:
        if f not in feat:
            feat[f] = 0.0

    return feat

# ─────────────────────────────────────────────────────────────
# PACKET PARSER
# ─────────────────────────────────────────────────────────────
def parse_packet(pkt) -> dict | None:
    """Extract raw per-packet fields to be aggregated later into a flow."""
    if not pkt.haslayer(IP):
        return None
    ip  = pkt[IP]
    tcp = pkt[TCP] if pkt.haslayer(TCP) else None
    udp = pkt[UDP] if pkt.haslayer(UDP) else None

    src_mac  = pkt[Ether].src if pkt.haslayer(Ether) else None
    dst_mac  = pkt[Ether].dst if pkt.haslayer(Ether) else None
    pkt_len  = len(pkt)
    ip_len   = getattr(ip, "len", pkt_len)
    hdr_len  = getattr(ip, "ihl", 5) * 4
    pay_len  = max(0, ip_len - hdr_len)
    ip_flags = int(ip.flags) if hasattr(ip, "flags") else 0

    win = tcp.window if tcp else 0
    mss = 0
    if tcp and tcp.options:
        for opt in tcp.options:
            if opt[0] == "MSS":
                mss = opt[1]
                break

    tcp_flags_int = int(tcp.flags) if tcp else 0
    f_syn = bool(tcp and tcp.flags.S)
    f_ack = bool(tcp and tcp.flags.A)
    f_fin = bool(tcp and tcp.flags.F)
    f_rst = bool(tcp and tcp.flags.R)
    f_psh = bool(tcp and tcp.flags.P)
    f_urg = bool(tcp and tcp.flags.U)

    src_port = (tcp.sport if tcp else (udp.sport if udp else 0))
    dst_port = (tcp.dport if tcp else (udp.dport if udp else 0))
    protocol = ip.proto
    is_frag  = bool(ip.flags.MF or ip.frag != 0)

    if LOCAL_IPS:
        if ip.src in LOCAL_IPS:
            direction = "outbound"
        elif ip.dst in LOCAL_IPS:
            direction = "inbound"
        else:
            direction = "unknown"
    else:
        direction = "unknown"

    return {
        "src": ip.src, "dst": ip.dst,
        "src_mac": src_mac, "dst_mac": dst_mac,
        "ttl": ip.ttl, "window": win,
        "ip_len": ip_len, "ip_flags": ip_flags,
        "pkt_size": pkt_len, "hdr_len": hdr_len, "pay_len": pay_len,
        "mss": mss, "tcp_flags": tcp_flags_int,
        "flags_syn": f_syn, "flags_ack": f_ack, "flags_fin": f_fin,
        "flags_rst": f_rst, "flags_psh": f_psh, "flags_urg": f_urg,
        "src_port": src_port, "dst_port": dst_port,
        "protocol": protocol, "is_fragment": is_frag,
        "timestamp": time.time(), "direction": direction,
    }

# ─────────────────────────────────────────────────────────────
# FLOW KEY
# ─────────────────────────────────────────────────────────────
def flow_key(p: dict) -> tuple:
    """Normalized 5-tuple so both directions of a conversation land together."""
    a = (p["src"], p["src_port"])
    b = (p["dst"], p["dst_port"])
    proto = p["protocol"]
    return (*min(a, b), *max(a, b), proto)

# ─────────────────────────────────────────────────────────────
# IP ATTRIBUTION
# ─────────────────────────────────────────────────────────────
def attribute_ip(fkey: tuple, pkts: list) -> str:
    """Return the external endpoint of a flow for strike/block purposes."""
    ip_a, _, ip_b, _, _ = fkey
    if LOCAL_IPS:
        if ip_a in LOCAL_IPS and ip_b not in LOCAL_IPS:
            return ip_b
        if ip_b in LOCAL_IPS and ip_a not in LOCAL_IPS:
            return ip_a
    return pkts[0]["src"]

# ─────────────────────────────────────────────────────────────
# BLOCK / UNBLOCK ENGINE
# ─────────────────────────────────────────────────────────────
def block_ip(ip_addr: str):
    """Add iptables DROP rule and record unblock time."""
    with _lock:
        if ip_addr in blocked_ips:
            return
        blocked_ips[ip_addr] = (time.time() + BLOCK_DURATION) if BLOCK_DURATION > 0 else 0
        stats["blocked"] += 1

    try:
        subprocess.run(
            ["iptables", "-A", "INPUT", "-s", ip_addr, "-j", "DROP"],
            check=True, capture_output=True,
        )
        duration_str = f"{BLOCK_DURATION}s" if BLOCK_DURATION > 0 else "permanent"
        log.warning(f"🔥 BLOCKED {ip_addr} ({duration_str}) | total={stats['blocked']}")
    except subprocess.CalledProcessError as exc:
        log.error(f"iptables error for {ip_addr}: {exc.stderr.decode().strip()}")


def unblock_ip(ip_addr: str):
    """Remove iptables DROP rule."""
    try:
        subprocess.run(
            ["iptables", "-D", "INPUT", "-s", ip_addr, "-j", "DROP"],
            check=True, capture_output=True,
        )
        log.info(f"✅ UNBLOCKED {ip_addr} (block expired)")
    except subprocess.CalledProcessError:
        pass  # rule may have been removed manually
    with _lock:
        blocked_ips.pop(ip_addr, None)
        ip_strikes.pop(ip_addr, None)


def check_unblocks():
    """Background thread: expire timed blocks and decay idle strike counters."""
    while not _shutdown.is_set():
        now = time.time()
        with _lock:
            expired = [ip for ip, until in blocked_ips.items() if until and until <= now]
        for ip in expired:
            unblock_ip(ip)

        # Decay strikes for IPs that haven't triggered recently
        with _lock:
            for ip in list(ip_strikes.keys()):
                if ip not in blocked_ips and ip_strikes[ip] > 0:
                    ip_strikes[ip] -= 1
                    if ip_strikes[ip] == 0:
                        del ip_strikes[ip]

        _shutdown.wait(timeout=STRIKE_DECAY_IV)

# ─────────────────────────────────────────────────────────────
# EVENT LOGGER  →  atomic write to events.json
# ─────────────────────────────────────────────────────────────
def write_event(src_ip: str, prob: float, label: str, action: str, strike: int):
    event = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "src_ip":    src_ip,
        "prob":      round(float(prob), 4),
        "label":     label,
        "action":    action,
        "strike":    strike,
        "stats":     dict(stats),
    }
    with _lock:
        event_buffer.append(event)
        snapshot = list(event_buffer)

    # Atomic rename: write to temp file then rename so dashboard never reads partial JSON
    dir_ = os.path.dirname(JSON_LOG_PATH)
    try:
        with tempfile.NamedTemporaryFile("w", dir=dir_, delete=False, suffix=".tmp") as tf:
            json.dump(snapshot, tf, indent=2)
            tmp_path = tf.name
        os.replace(tmp_path, JSON_LOG_PATH)
    except Exception as exc:
        log.error(f"JSON write error: {exc}")

    log.info(f"SRC={src_ip} p={prob:.4f} {label} strike={strike} action={action}")

# ─────────────────────────────────────────────────────────────
# FLOW PROCESSOR  (background thread, runs every FLOW_WINDOW seconds)
# ─────────────────────────────────────────────────────────────
def process_flows():
    """Drain the flow buffer, run inference, apply strike/block logic."""
    while not _shutdown.is_set():
        _shutdown.wait(timeout=FLOW_WINDOW)

        with _lock:
            snapshot = dict(flows)
            flows.clear()

        if not snapshot:
            continue

        for fkey, pkts in snapshot.items():
            if len(pkts) < MIN_FLOW_PKTS:
                continue

            src_ip = attribute_ip(fkey, pkts)

            with _lock:
                already_blocked = src_ip in blocked_ips
            if already_blocked:
                continue

            feat_dict = aggregate_flow(pkts)
            if not feat_dict:
                continue

            try:
                df    = pd.DataFrame([feat_dict])[FEATURES]
                prob  = float(model.predict_proba(df)[0][1])
            except Exception as exc:
                log.error(f"Inference error for {src_ip}: {exc}")
                continue

            label = "ATTACK" if prob >= THRESHOLD else "BENIGN"

            with _lock:
                stats["total"] += 1
                if label == "ATTACK":
                    stats["attacks"] += 1
                    ip_strikes[src_ip] += 1
                    strike = ip_strikes[src_ip]
                else:
                    stats["benign"] += 1
                    ip_strikes[src_ip] = max(0, ip_strikes[src_ip] - 1)
                    strike = ip_strikes[src_ip]

            if label == "ATTACK":
                if strike >= STRIKE_LIMIT:
                    block_ip(src_ip)
                    action = "BLOCKED (iptables DROP)"
                else:
                    action = f"Strike {strike}/{STRIKE_LIMIT}"
            else:
                action = "Allow"

            write_event(src_ip, prob, label, action, strike)

    # Graceful shutdown: flush any remaining packets
    log.info("Flow processor shutting down — flushing remaining flows ...")
    with _lock:
        snapshot = dict(flows)
        flows.clear()
    for fkey, pkts in snapshot.items():
        if len(pkts) >= MIN_FLOW_PKTS:
            src_ip = attribute_ip(fkey, pkts)
            feat_dict = aggregate_flow(pkts)
            if feat_dict:
                try:
                    df   = pd.DataFrame([feat_dict])[FEATURES]
                    prob = float(model.predict_proba(df)[0][1])
                    label = "ATTACK" if prob >= THRESHOLD else "BENIGN"
                    write_event(src_ip, prob, label, "shutdown-flush", 0)
                except Exception:
                    pass

# ─────────────────────────────────────────────────────────────
# PACKET CALLBACK
# ─────────────────────────────────────────────────────────────
def packet_callback(pkt):
    p = parse_packet(pkt)
    if p is None:
        return
    src, dst = p["src"], p["dst"]

    with _lock:
        if src in blocked_ips:
            return

    if src in ALLOWLIST_IPS or dst in ALLOWLIST_IPS:
        return
    if p["src_port"] in ALLOWLIST_PORTS or p["dst_port"] in ALLOWLIST_PORTS:
        return

    # Ignore purely local-to-local traffic (e.g. loopback or inter-container)
    if LOCAL_IPS and src in LOCAL_IPS and dst in LOCAL_IPS:
        return

    with _lock:
        flows[flow_key(p)].append(p)

# ─────────────────────────────────────────────────────────────
# SIGNAL HANDLER
# ─────────────────────────────────────────────────────────────
def _handle_signal(signum, frame):
    log.info(f"Received signal {signum} — initiating graceful shutdown ...")
    _shutdown.set()

# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main():
    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT,  _handle_signal)

    log.info("=" * 60)
    log.info("  THREATSHIELD v2.0 — AI Intrusion Detection & Prevention")
    log.info("=" * 60)
    log.info(f"  threshold={THRESHOLD}  strike_limit={STRIKE_LIMIT}  "
             f"flow_window={FLOW_WINDOW}s  min_pkts={MIN_FLOW_PKTS}")
    log.info(f"  block_duration={'permanent' if BLOCK_DURATION==0 else f'{BLOCK_DURATION}s'}  "
             f"local_ips={LOCAL_IPS or '(not set)'}")

    load_model()

    iface = detect_interface()

    # Flow processor thread
    fp = threading.Thread(target=process_flows, daemon=True, name="FlowProcessor")
    fp.start()

    # Unblock / strike-decay thread
    ub = threading.Thread(target=check_unblocks, daemon=True, name="UnblockWatcher")
    ub.start()

    log.info(f"✅ Threads started | sniffing on {iface}")
    log.info(f"📄 Events: {JSON_LOG_PATH}")

    conf.verb = 0
    try:
        sniff(iface=iface, prn=packet_callback, store=False,
              stop_filter=lambda _: _shutdown.is_set())
    except Exception as exc:
        log.error(f"Sniff error: {exc}")
        _shutdown.set()

    fp.join(timeout=15)
    log.info("THREATSHIELD shut down cleanly.")


if __name__ == "__main__":
    main()
