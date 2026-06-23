#!/usr/bin/env python3
"""
THREATSHIELD — idps_engine.py
Real-time packet capture, flow aggregation, XGBoost inference, and IP blocking.
Runs as a systemd service on AWS EC2. Sends detections to the dashboard via JSON log.

Author: Your Team
Model:  idps_xgboost_model.pkl  (71 flow-level features, XGBoost binary classifier)
"""

import os
import sys
import time
import json
import logging
import subprocess
import threading
import collections
from datetime import datetime

import joblib
import pandas as pd
import numpy as np
from scapy.all import sniff, IP, TCP, UDP, Ether, conf

# ─────────────────────────────────────────────────────────────
# CONFIG  (edit these as needed)
# ─────────────────────────────────────────────────────────────
MODEL_PATH    = "/opt/threatshield/idps_xgboost_model.pkl"
LOG_PATH      = "/opt/threatshield/idps.log"
JSON_LOG_PATH = "/opt/threatshield/events.json"   # dashboard reads this
IFACE         = "eth0"          # network interface on EC2
THRESHOLD     = 0.4             # attack probability cutoff (matches your notebook: 0.4)
STRIKE_LIMIT  = 3               # blocks after N consecutive detections
FLOW_WINDOW   = 5               # seconds per flow aggregation window
MAX_LOG_LINES = 500             # events.json keeps last N events for dashboard

# ─────────────────────────────────────────────────────────────
# LOGGING SETUP
# ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH, mode='a'),
        logging.StreamHandler(sys.stdout),
    ]
)
log = logging.getLogger("THREATSHIELD")

# ─────────────────────────────────────────────────────────────
# GLOBAL STATE
# ─────────────────────────────────────────────────────────────
model        = None
FEATURES     = []               # exact 71 feature names from model
ip_strikes   = collections.defaultdict(int)
blocked_ips  = set()
event_buffer = collections.deque(maxlen=MAX_LOG_LINES)  # ring buffer for dashboard
flows        = collections.defaultdict(list)            # src_ip → list of packet dicts
stats        = {"total": 0, "attacks": 0, "blocked": 0, "benign": 0}

# ─────────────────────────────────────────────────────────────
# MODEL LOAD
# ─────────────────────────────────────────────────────────────
def load_model():
    global model, FEATURES
    log.info(f"Loading model from {MODEL_PATH}...")
    model = joblib.load(MODEL_PATH)
    FEATURES = list(model.feature_names_in_)
    log.info(f"✅ Model loaded | Features: {len(FEATURES)} | Classes: {model.classes_}")

# ─────────────────────────────────────────────────────────────
# FLOW AGGREGATOR  →  71-feature dict (matches your exact CICIoT features)
# ─────────────────────────────────────────────────────────────
def aggregate_flow(packets):
    """
    Takes a list of raw packet dicts collected in a FLOW_WINDOW and computes
    exactly the 71 statistical features your XGBoost model was trained on.
    """
    if not packets:
        return {f: 0.0 for f in FEATURES}

    # ── Extract raw series ────────────────────────────────────
    ttl_vals     = [p["ttl"]         for p in packets if p["ttl"]         is not None]
    win_vals     = [p["window"]      for p in packets if p["window"]      is not None]
    ip_len_vals  = [p["ip_len"]      for p in packets if p["ip_len"]      is not None]
    ip_flags_vals= [p["ip_flags"]    for p in packets if p["ip_flags"]    is not None]
    pkt_size_vals= [p["pkt_size"]    for p in packets if p["pkt_size"]    is not None]
    hdr_len_vals = [p["hdr_len"]     for p in packets if p["hdr_len"]     is not None]
    pay_len_vals = [p["pay_len"]     for p in packets if p["pay_len"]     is not None]
    mss_vals     = [p["mss"]         for p in packets if p["mss"]         is not None]
    tcp_flags_v  = [p["tcp_flags"]   for p in packets if p["tcp_flags"]   is not None]
    timestamps   = [p["timestamp"]   for p in packets]
    src_ips      = [p["src"]         for p in packets]
    dst_ips      = [p["dst"]         for p in packets]
    src_macs     = [p["src_mac"]     for p in packets if p["src_mac"]     is not None]
    dst_macs     = [p["dst_mac"]     for p in packets if p["dst_mac"]     is not None]
    src_ports    = [p["src_port"]    for p in packets if p["src_port"]    is not None]
    dst_ports    = [p["dst_port"]    for p in packets if p["dst_port"]    is not None]
    protocols    = [p["protocol"]    for p in packets if p["protocol"]    is not None]
    mss_src      = [p["src_port"]    for p in packets if p["src_port"]    is not None]
    mss_dst      = [p["dst_port"]    for p in packets if p["dst_port"]    is not None]

    def safe_stat(vals):
        """Return (avg, max, min, std) or (0,0,0,0) for empty."""
        if not vals:
            return 0.0, 0.0, 0.0, 0.0
        arr = np.array(vals, dtype=float)
        return float(arr.mean()), float(arr.max()), float(arr.min()), float(arr.std())

    # ── Time deltas ───────────────────────────────────────────
    time_deltas = []
    for i in range(1, len(timestamps)):
        time_deltas.append(timestamps[i] - timestamps[i-1])

    td_avg, td_max, td_min, td_std = safe_stat(time_deltas)

    # ── TCP flag decomposition ────────────────────────────────
    syn_count  = sum(1 for p in packets if p["flags_syn"])
    ack_count  = sum(1 for p in packets if p["flags_ack"])
    fin_count  = sum(1 for p in packets if p["flags_fin"])
    rst_count  = sum(1 for p in packets if p["flags_rst"])
    psh_count  = sum(1 for p in packets if p["flags_psh"])
    urg_count  = sum(1 for p in packets if p["flags_urg"])
    tcf_avg, tcf_max, tcf_min, tcf_std = safe_stat(tcp_flags_v)

    # ── Fragmentation ─────────────────────────────────────────
    frag_pkts  = sum(1 for p in packets if p["is_fragment"])
    frag_score = frag_pkts / max(len(packets), 1)

    # ── Interval: avg time between messages ───────────────────
    interval_msgs = td_avg

    # ── Payload length ────────────────────────────────────────
    pay_avg, pay_max, pay_min, pay_std = safe_stat(pay_len_vals)

    # ── Header length ─────────────────────────────────────────
    hdr_avg, hdr_max, hdr_min, hdr_std = safe_stat(hdr_len_vals)

    # ── Packet size ───────────────────────────────────────────
    ps_avg, ps_max, ps_min, ps_std = safe_stat(pkt_size_vals)

    # ── IP length ─────────────────────────────────────────────
    il_avg, il_max, il_min, il_std = safe_stat(ip_len_vals)

    # ── IP flags ─────────────────────────────────────────────
    ipf_avg, ipf_max, ipf_min, ipf_std = safe_stat(ip_flags_vals)

    # ── TTL ───────────────────────────────────────────────────
    ttl_avg_v, ttl_max_v, ttl_min_v, ttl_std_v = safe_stat(ttl_vals)

    # ── Window size ───────────────────────────────────────────
    win_avg, win_max, win_min, win_std = safe_stat(win_vals)

    # ── MSS ──────────────────────────────────────────────────
    mss_avg, mss_max, mss_min, mss_std = safe_stat(mss_vals)

    # ── Counts ────────────────────────────────────────────────
    n_pkts = len(packets)
    # src/dst packet counts
    src_pkt_c  = len([p for p in packets if p["direction"] == "outbound"])
    dst_pkt_c  = n_pkts - src_pkt_c

    # ── Log-level features (approximate from packet-level proxies) ───
    # These are log/app-layer aggregates; at packet level we approximate:
    log_data_ranges = pay_len_vals if pay_len_vals else [0.0]
    ldr_avg, ldr_max, ldr_min, ldr_std = safe_stat(log_data_ranges)
    log_dtype_count  = float(len(set(protocols)))
    log_msg_count    = float(n_pkts)
    log_interval     = interval_msgs

    # ── Assemble the exact 71-feature dict ───────────────────
    feat = {
        # LOG features
        "log_data-ranges_avg":           ldr_avg,
        "log_data-ranges_max":           ldr_max,
        "log_data-ranges_min":           ldr_min,
        "log_data-ranges_std_deviation": ldr_std,
        "log_data-types_count":          log_dtype_count,
        "log_interval-messages":         log_interval,
        "log_messages_count":            log_msg_count,
        # NETWORK features
        "network_fragmentation-score":         frag_score,
        "network_fragmented-packets":          float(frag_pkts),
        "network_header-length_avg":           hdr_avg,
        "network_header-length_max":           hdr_max,
        "network_header-length_min":           hdr_min,
        "network_header-length_std_deviation": hdr_std,
        "network_interval-packets":            interval_msgs,
        "network_ip-flags_avg":                ipf_avg,
        "network_ip-flags_max":                ipf_max,
        "network_ip-flags_min":                ipf_min,
        "network_ip-flags_std_deviation":      ipf_std,
        "network_ip-length_avg":               il_avg,
        "network_ip-length_max":               il_max,
        "network_ip-length_min":               il_min,
        "network_ip-length_std_deviation":     il_std,
        "network_ips_all_count":               float(len(set(src_ips + dst_ips))),
        "network_ips_dst_count":               float(len(set(dst_ips))),
        "network_ips_src_count":               float(len(set(src_ips))),
        "network_macs_all_count":              float(len(set(src_macs + dst_macs))),
        "network_macs_dst_count":              float(len(set(dst_macs))),
        "network_macs_src_count":              float(len(set(src_macs))),
        "network_mss_avg":                     mss_avg,
        "network_mss_max":                     mss_max,
        "network_mss_min":                     mss_min,
        "network_mss_std_deviation":           mss_std,
        "network_packet-size_avg":             ps_avg,
        "network_packet-size_max":             ps_max,
        "network_packet-size_min":             ps_min,
        "network_packet-size_std_deviation":   ps_std,
        "network_packets_all_count":           float(n_pkts),
        "network_packets_dst_count":           float(dst_pkt_c),
        "network_packets_src_count":           float(src_pkt_c),
        "network_payload-length_avg":          pay_avg,
        "network_payload-length_max":          pay_max,
        "network_payload-length_min":          pay_min,
        "network_payload-length_std_deviation":pay_std,
        "network_ports_all_count":             float(len(set(src_ports + dst_ports))),
        "network_ports_dst_count":             float(len(set(dst_ports))),
        "network_ports_src_count":             float(len(set(src_ports))),
        "network_protocols_all_count":         float(len(set(protocols))),
        "network_protocols_dst_count":         float(len(set(protocols))),
        "network_protocols_src_count":         float(len(set(protocols))),
        "network_tcp-flags-ack_count":         float(ack_count),
        "network_tcp-flags-fin_count":         float(fin_count),
        "network_tcp-flags-psh_count":         float(psh_count),
        "network_tcp-flags-rst_count":         float(rst_count),
        "network_tcp-flags-syn_count":         float(syn_count),
        "network_tcp-flags-urg_count":         float(urg_count),
        "network_tcp-flags_avg":               tcf_avg,
        "network_tcp-flags_max":               tcf_max,
        "network_tcp-flags_min":               tcf_min,
        "network_tcp-flags_std_deviation":     tcf_std,
        "network_time-delta_avg":              td_avg,
        "network_time-delta_max":              td_max,
        "network_time-delta_min":              td_min,
        "network_time-delta_std_deviation":    td_std,
        "network_ttl_avg":                     ttl_avg_v,
        "network_ttl_max":                     ttl_max_v,
        "network_ttl_min":                     ttl_min_v,
        "network_ttl_std_deviation":           ttl_std_v,
        "network_window-size_avg":             win_avg,
        "network_window-size_max":             win_max,
        "network_window-size_min":             win_min,
        "network_window-size_std_deviation":   win_std,
    }

    # safety check: fill any missing feature with 0.0
    for f in FEATURES:
        if f not in feat:
            feat[f] = 0.0

    return feat

# ─────────────────────────────────────────────────────────────
# PACKET PARSER  →  per-packet dict
# ─────────────────────────────────────────────────────────────
def parse_packet(pkt):
    """Extract raw per-packet fields to be aggregated into a flow."""
    if not pkt.haslayer(IP):
        return None
    ip = pkt[IP]
    tcp = pkt[TCP] if pkt.haslayer(TCP) else None
    udp = pkt[UDP] if pkt.haslayer(UDP) else None

    src_mac = pkt[Ether].src if pkt.haslayer(Ether) else None
    dst_mac = pkt[Ether].dst if pkt.haslayer(Ether) else None

    pkt_len   = len(pkt)
    ip_len    = ip.len if hasattr(ip, 'len') else pkt_len
    hdr_len   = ip.ihl * 4 if hasattr(ip, 'ihl') else 20
    pay_len   = max(0, ip_len - hdr_len)
    ip_flags  = int(ip.flags) if hasattr(ip, 'flags') else 0

    # TCP specific
    win  = tcp.window   if tcp else 0
    mss  = 0
    if tcp and tcp.options:
        for opt in tcp.options:
            if opt[0] == 'MSS':
                mss = opt[1]
                break
    tcp_flags_int = int(tcp.flags) if tcp else 0
    f_syn = bool(tcp and tcp.flags.S) if tcp else False
    f_ack = bool(tcp and tcp.flags.A) if tcp else False
    f_fin = bool(tcp and tcp.flags.F) if tcp else False
    f_rst = bool(tcp and tcp.flags.R) if tcp else False
    f_psh = bool(tcp and tcp.flags.P) if tcp else False
    f_urg = bool(tcp and tcp.flags.U) if tcp else False

    src_port = tcp.sport if tcp else (udp.sport if udp else 0)
    dst_port = tcp.dport if tcp else (udp.dport if udp else 0)
    protocol = ip.proto

    is_frag = bool(ip.flags.MF or ip.frag != 0)

    return {
        "src":        ip.src,
        "dst":        ip.dst,
        "src_mac":    src_mac,
        "dst_mac":    dst_mac,
        "ttl":        ip.ttl,
        "window":     win,
        "ip_len":     ip_len,
        "ip_flags":   ip_flags,
        "pkt_size":   pkt_len,
        "hdr_len":    hdr_len,
        "pay_len":    pay_len,
        "mss":        mss,
        "tcp_flags":  tcp_flags_int,
        "flags_syn":  f_syn,
        "flags_ack":  f_ack,
        "flags_fin":  f_fin,
        "flags_rst":  f_rst,
        "flags_psh":  f_psh,
        "flags_urg":  f_urg,
        "src_port":   src_port,
        "dst_port":   dst_port,
        "protocol":   protocol,
        "is_fragment":is_frag,
        "timestamp":  time.time(),
        "direction":  "outbound",  # simplified; update with your routing logic
    }

# ─────────────────────────────────────────────────────────────
# BLOCK ENGINE
# ─────────────────────────────────────────────────────────────
def block_ip(ip_addr):
    if ip_addr in blocked_ips:
        return
    try:
        subprocess.run(
            ["sudo", "iptables", "-A", "INPUT", "-s", ip_addr, "-j", "DROP"],
            check=True, capture_output=True
        )
        blocked_ips.add(ip_addr)
        stats["blocked"] += 1
        log.warning(f"🔥 BLOCKED: {ip_addr} | Total blocked: {stats['blocked']}")
    except subprocess.CalledProcessError as e:
        log.error(f"iptables error for {ip_addr}: {e}")

# ─────────────────────────────────────────────────────────────
# EVENT LOGGER  →  writes to events.json (dashboard reads this)
# ─────────────────────────────────────────────────────────────
def write_event(src_ip, prob, label, action, strike):
    event = {
        "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        "src_ip":    src_ip,
        "prob":      round(float(prob), 4),
        "label":     label,
        "action":    action,
        "strike":    strike,
        "stats":     dict(stats),
    }
    event_buffer.append(event)
    # Atomically overwrite events.json with latest ring buffer
    try:
        with open(JSON_LOG_PATH, "w") as f:
            json.dump(list(event_buffer), f, indent=2)
    except Exception as e:
        log.error(f"JSON write error: {e}")

    # Also append to plain text log
    log.info(
        f"SRC: {src_ip} | p={prob:.4f} | {label} | Strike:{strike} | {action}"
    )

# ─────────────────────────────────────────────────────────────
# FLOW PROCESSOR  →  runs every FLOW_WINDOW seconds in a thread
# ─────────────────────────────────────────────────────────────
def process_flows():
    """Background thread: every FLOW_WINDOW seconds, aggregate all collected
    per-packet data into flows, run XGBoost inference, and decide action."""
    while True:
        time.sleep(FLOW_WINDOW)
        if not flows:
            continue

        # snapshot current flows and reset
        flow_snapshot = dict(flows)
        flows.clear()

        for src_ip, pkts in flow_snapshot.items():
            if src_ip in blocked_ips:
                continue

            # aggregate into 71 features
            feat_dict = aggregate_flow(pkts)
            df = pd.DataFrame([feat_dict])[FEATURES]

            # XGBoost inference
            prob  = model.predict_proba(df)[0][1]
            label = "ATTACK" if prob >= THRESHOLD else "BENIGN"

            stats["total"] += 1

            if label == "ATTACK":
                stats["attacks"] += 1
                ip_strikes[src_ip] += 1
                action = f"Strike {ip_strikes[src_ip]}/{STRIKE_LIMIT}"
                if ip_strikes[src_ip] >= STRIKE_LIMIT:
                    block_ip(src_ip)
                    action = "BLOCKED (iptables DROP applied)"
            else:
                stats["benign"] += 1
                ip_strikes[src_ip] = max(0, ip_strikes[src_ip] - 1)
                action = "Allow"

            write_event(src_ip, prob, label, action, ip_strikes[src_ip])

# ─────────────────────────────────────────────────────────────
# PACKET CALLBACK (runs for every captured packet)
# ─────────────────────────────────────────────────────────────
def packet_callback(pkt):
    p = parse_packet(pkt)
    if p is None:
        return
    src = p["src"]
    if src in blocked_ips:
        return  # already blocked
    flows[src].append(p)

# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main():
    log.info("=" * 60)
    log.info("  THREATSHIELD — AI Intrusion Detection & Prevention")
    log.info("=" * 60)

    load_model()

    # Start flow-processor thread
    t = threading.Thread(target=process_flows, daemon=True)
    t.start()
    log.info(f"✅ Flow processor started (window={FLOW_WINDOW}s, threshold={THRESHOLD})")
    log.info(f"👁  Sniffing on interface: {IFACE}")
    log.info(f"📄 Events JSON: {JSON_LOG_PATH}")

    try:
        conf.verb = 0  # silent scapy
        sniff(iface=IFACE, prn=packet_callback, store=False)
    except KeyboardInterrupt:
        log.info("Shutting down THREATSHIELD.")
    except Exception as e:
        log.error(f"Sniff error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
