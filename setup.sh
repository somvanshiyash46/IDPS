#!/bin/bash
# ============================================================
# THREATSHIELD v2.0 — AWS EC2 Setup Script
# Tested on Ubuntu 22.04 LTS (t3.small or larger recommended).
#
# Usage:
#   bash setup.sh            # full install
#   bash setup.sh --uninstall  # remove services, venv, and files
#
# The script is idempotent: safe to re-run after partial failures.
# On any unexpected error it stops immediately (set -e) and prints
# the failed command so you can fix and re-run.
# ============================================================

set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
info()    { echo -e "${BLUE}[INFO]${NC}  $1"; }
success() { echo -e "${GREEN}[OK]${NC}   $1"; }
warn()    { echo -e "${YELLOW}[WARN]${NC} $1"; }
error()   { echo -e "${RED}[ERR]${NC}  $1" >&2; exit 1; }

INSTALL_DIR="/opt/threatshield"
VENV="$INSTALL_DIR/venv"
UPLOAD_DIR="/home/ubuntu"

# ─────────────────────────────────────────────────────────────
# UNINSTALL MODE
# ─────────────────────────────────────────────────────────────
if [[ "${1:-}" == "--uninstall" ]]; then
  echo -e "${YELLOW}======================================${NC}"
  echo -e "${YELLOW}   THREATSHIELD — Uninstall${NC}"
  echo -e "${YELLOW}======================================${NC}"
  info "Stopping and disabling services..."
  sudo systemctl stop  idps idps-dashboard 2>/dev/null || true
  sudo systemctl disable idps idps-dashboard 2>/dev/null || true
  sudo rm -f /etc/systemd/system/idps.service /etc/systemd/system/idps-dashboard.service
  sudo systemctl daemon-reload
  info "Removing install directory..."
  sudo rm -rf "$INSTALL_DIR"
  info "Removing logrotate config..."
  sudo rm -f /etc/logrotate.d/threatshield
  success "THREATSHIELD uninstalled. iptables rules (if any) must be cleared manually:"
  echo "  sudo iptables -F INPUT"
  exit 0
fi

# ─────────────────────────────────────────────────────────────
# BANNER
# ─────────────────────────────────────────────────────────────
echo ""
echo -e "${BLUE}============================================================${NC}"
echo -e "${BLUE}   THREATSHIELD v2.0 — AWS EC2 Automated Setup Script${NC}"
echo -e "${BLUE}============================================================${NC}"
echo ""

# Must run as ubuntu (not root) so the venv is usable by both services
if [[ "$(whoami)" == "root" ]]; then
  error "Run as the 'ubuntu' user, not root. sudo is invoked internally where needed."
fi

# ─────────────────────────────────────────────────────────────
# STEP 1: System update
# ─────────────────────────────────────────────────────────────
info "STEP 1 — Updating system packages..."
sudo apt-get update -qq
sudo apt-get upgrade -y -qq
success "System updated"

# ─────────────────────────────────────────────────────────────
# STEP 2: System libraries
# ─────────────────────────────────────────────────────────────
info "STEP 2 — Installing system libraries..."
sudo apt-get install -y -qq \
    python3 python3-pip python3-venv \
    libpcap-dev tcpdump \
    iptables iptables-persistent ufw \
    logrotate \
    git curl wget unzip net-tools
success "System libraries installed"

# ─────────────────────────────────────────────────────────────
# STEP 3: Install directory
# ─────────────────────────────────────────────────────────────
info "STEP 3 — Creating $INSTALL_DIR ..."
sudo mkdir -p "$INSTALL_DIR"
sudo chown ubuntu:ubuntu "$INSTALL_DIR"
success "$INSTALL_DIR ready"

# ─────────────────────────────────────────────────────────────
# STEP 4: Python virtual environment
# ─────────────────────────────────────────────────────────────
info "STEP 4 — Creating Python venv..."
if [[ ! -f "$VENV/bin/python3" ]]; then
  python3 -m venv "$VENV"
  success "Venv created at $VENV"
else
  success "Venv already exists — skipping"
fi

# ─────────────────────────────────────────────────────────────
# STEP 5: Python packages
# ─────────────────────────────────────────────────────────────
info "STEP 5 — Installing Python packages (this takes 2–3 min)..."
"$VENV/bin/pip" install --quiet --upgrade pip
"$VENV/bin/pip" install --quiet \
    xgboost \
    scikit-learn \
    pandas \
    numpy \
    scapy \
    joblib \
    flask \
    pyarrow
success "Python packages installed"

# ─────────────────────────────────────────────────────────────
# STEP 6: Copy project files
# ─────────────────────────────────────────────────────────────
info "STEP 6 — Copying project files..."

copy_if_exists() {
  local src="$UPLOAD_DIR/$1"
  local dst="$INSTALL_DIR/$1"
  if [[ -f "$src" ]]; then
    cp "$src" "$dst"
    success "$1 → $INSTALL_DIR/"
  else
    warn "$1 not found at $src — skipping"
  fi
}

copy_if_exists "idps_engine.py"
copy_if_exists "dashboard.py"

# Model file check
if [[ -f "$UPLOAD_DIR/idps_xgboost_model.pkl" ]]; then
  cp "$UPLOAD_DIR/idps_xgboost_model.pkl" "$INSTALL_DIR/"
  success "idps_xgboost_model.pkl copied"
else
  warn "Model file not found at $UPLOAD_DIR/idps_xgboost_model.pkl"
  warn "SCP it manually after setup:"
  warn "  scp idps_xgboost_model.pkl ubuntu@YOUR_EC2_IP:/home/ubuntu/"
  warn "Then: sudo systemctl start idps"
fi

# ─────────────────────────────────────────────────────────────
# STEP 7: Initialise log files
# ─────────────────────────────────────────────────────────────
info "STEP 7 — Initialising log files..."
touch "$INSTALL_DIR/idps.log" "$INSTALL_DIR/dashboard.log"
[[ -f "$INSTALL_DIR/events.json" ]] || echo "[]" > "$INSTALL_DIR/events.json"
chmod 664 "$INSTALL_DIR/events.json" "$INSTALL_DIR/idps.log" "$INSTALL_DIR/dashboard.log"
success "Log files ready"

# ─────────────────────────────────────────────────────────────
# STEP 8: Logrotate
# ─────────────────────────────────────────────────────────────
info "STEP 8 — Configuring logrotate..."
sudo tee /etc/logrotate.d/threatshield > /dev/null <<EOF
$INSTALL_DIR/*.log {
    daily
    rotate 7
    compress
    missingok
    notifempty
    copytruncate
}
EOF
success "Logrotate configured (daily, 7-day retention)"

# ─────────────────────────────────────────────────────────────
# STEP 9: Detect instance IPs
# ─────────────────────────────────────────────────────────────
info "STEP 9 — Detecting instance IP addresses..."
META="http://169.254.169.254/latest/meta-data"
LOCAL_PRIVATE_IP=$(curl -s --max-time 3 "$META/local-ipv4"  2>/dev/null || true)
LOCAL_PUBLIC_IP=$(curl -s  --max-time 3 "$META/public-ipv4" 2>/dev/null || true)

# Build comma-separated list, strip leading/trailing commas
TS_LOCAL_IPS="${LOCAL_PRIVATE_IP:+$LOCAL_PRIVATE_IP,}${LOCAL_PUBLIC_IP}"
TS_LOCAL_IPS="${TS_LOCAL_IPS%,}"
TS_LOCAL_IPS="${TS_LOCAL_IPS#,}"

if [[ -z "$TS_LOCAL_IPS" ]]; then
  warn "Could not auto-detect IPs from EC2 metadata."
  warn "Set THREATSHIELD_LOCAL_IPS manually in /etc/systemd/system/idps.service"
  TS_LOCAL_IPS=""
else
  success "Detected IPs: $TS_LOCAL_IPS"
fi

# ─────────────────────────────────────────────────────────────
# STEP 10: systemd — IDPS engine
# ─────────────────────────────────────────────────────────────
info "STEP 10 — Creating systemd service: idps..."
sudo tee /etc/systemd/system/idps.service > /dev/null <<EOF
[Unit]
Description=THREATSHIELD AI-IDPS Engine v2.0
After=network.target
Wants=network.target

[Service]
Type=simple
User=root
WorkingDirectory=$INSTALL_DIR
ExecStart=$VENV/bin/python3 $INSTALL_DIR/idps_engine.py
Restart=on-failure
RestartSec=5
TimeoutStopSec=15
StandardOutput=append:$INSTALL_DIR/idps.log
StandardError=append:$INSTALL_DIR/idps.log
Environment=PYTHONUNBUFFERED=1
Environment=THREATSHIELD_LOCAL_IPS=$TS_LOCAL_IPS

# Tunable knobs — edit and run: sudo systemctl daemon-reload && sudo systemctl restart idps
Environment=TS_THRESHOLD=0.75
Environment=TS_STRIKE_LIMIT=2
Environment=TS_FLOW_WINDOW=5
Environment=TS_MIN_FLOW_PKTS=3
Environment=TS_BLOCK_DURATION=600
Environment=TS_STRIKE_DECAY=60

[Install]
WantedBy=multi-user.target
EOF
success "idps.service written"

# ─────────────────────────────────────────────────────────────
# STEP 11: systemd — Dashboard
# ─────────────────────────────────────────────────────────────
info "STEP 11 — Creating systemd service: idps-dashboard..."
sudo tee /etc/systemd/system/idps-dashboard.service > /dev/null <<EOF
[Unit]
Description=THREATSHIELD Live Dashboard v2.0
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=$INSTALL_DIR
ExecStart=$VENV/bin/python3 $INSTALL_DIR/dashboard.py
Restart=on-failure
RestartSec=5
StandardOutput=append:$INSTALL_DIR/dashboard.log
StandardError=append:$INSTALL_DIR/dashboard.log
Environment=PYTHONUNBUFFERED=1
Environment=TS_JSON_PATH=$INSTALL_DIR/events.json
Environment=TS_THRESHOLD=0.75
Environment=TS_STRIKE_LIMIT=2

[Install]
WantedBy=multi-user.target
EOF
success "idps-dashboard.service written"

# ─────────────────────────────────────────────────────────────
# STEP 12: Enable and start services
# ─────────────────────────────────────────────────────────────
info "STEP 12 — Enabling services..."
sudo systemctl daemon-reload
sudo systemctl enable idps idps-dashboard

info "Starting dashboard..."
sudo systemctl restart idps-dashboard
success "Dashboard service started"

if [[ -f "$INSTALL_DIR/idps_xgboost_model.pkl" ]]; then
  info "Starting IDPS engine..."
  sudo systemctl restart idps
  success "IDPS engine started"
else
  warn "IDPS engine NOT started — model file missing."
  warn "After copying the model, run:  sudo systemctl start idps"
fi

# ─────────────────────────────────────────────────────────────
# STEP 13: Firewall
# ─────────────────────────────────────────────────────────────
info "STEP 13 — Configuring UFW firewall..."
sudo ufw --force enable
sudo ufw allow ssh          # port 22 — keep SSH open!
sudo ufw allow 5000/tcp     # dashboard
sudo ufw allow 80/tcp       # optional HTTP
sudo ufw allow 5001/tcp     # custom sensor simulator
success "Firewall: SSH(22), Dashboard(5000), Sensor(5001) allowed"

# Persist existing iptables rules so they survive reboots
sudo netfilter-persistent save 2>/dev/null || true

# ─────────────────────────────────────────────────────────────
# DONE
# ─────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}============================================================${NC}"
echo -e "${GREEN}   ✅  THREATSHIELD SETUP COMPLETE!${NC}"
echo -e "${GREEN}============================================================${NC}"
echo ""

PUBLIC_IP=$(curl -s --max-time 5 http://checkip.amazonaws.com 2>/dev/null || echo "YOUR_EC2_PUBLIC_IP")

echo -e "${YELLOW}📊  Dashboard:${NC}      http://$PUBLIC_IP:5000"
echo -e "${YELLOW}📄  IDPS log:${NC}       $INSTALL_DIR/idps.log"
echo -e "${YELLOW}📄  Events JSON:${NC}    $INSTALL_DIR/events.json"
echo -e "${YELLOW}📄  Dashboard log:${NC}  $INSTALL_DIR/dashboard.log"
echo ""
echo -e "${BLUE}Useful commands:${NC}"
echo "  sudo systemctl status  idps               # engine status"
echo "  sudo systemctl status  idps-dashboard     # dashboard status"
echo "  sudo systemctl restart idps               # restart engine"
echo "  tail -f $INSTALL_DIR/idps.log             # live engine log"
echo "  sudo iptables -L INPUT -n --line-numbers  # view blocked IPs"
echo "  sudo iptables -D INPUT <line-number>      # manually unblock"
echo "  bash setup.sh --uninstall                 # remove everything"
echo ""
