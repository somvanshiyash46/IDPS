#!/bin/bash
# ============================================================
# THREATSHIELD — AWS EC2 Setup Script
# Run this script on your Ubuntu 22.04 EC2 instance as ubuntu user.
# Usage:  bash setup.sh
# ============================================================

set -e  # exit on any error

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
info()    { echo -e "${BLUE}[INFO]${NC} $1"; }
success() { echo -e "${GREEN}[✅ OK]${NC} $1"; }
warn()    { echo -e "${YELLOW}[WARN]${NC} $1"; }
error()   { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

echo ""
echo -e "${BLUE}========================================================${NC}"
echo -e "${BLUE}   THREATSHIELD — AWS EC2 Automated Setup Script${NC}"
echo -e "${BLUE}========================================================${NC}"
echo ""

# ─────────────────────────────────────────────────────────────
# STEP 1: System update
# ─────────────────────────────────────────────────────────────
info "STEP 1: Updating system packages..."
sudo apt-get update -qq
sudo apt-get upgrade -y -qq
success "System updated"

# ─────────────────────────────────────────────────────────────
# STEP 2: Install Python & system libs
# ─────────────────────────────────────────────────────────────
info "STEP 2: Installing Python 3, pip, and system libraries..."
sudo apt-get install -y -qq \
    python3 python3-pip python3-venv \
    libpcap-dev tcpdump \
    iptables ufw \
    git curl wget unzip net-tools
success "System libraries installed"

# ─────────────────────────────────────────────────────────────
# STEP 3: Create project directory
# ─────────────────────────────────────────────────────────────
info "STEP 3: Creating project directory /opt/threatshield ..."
sudo mkdir -p /opt/threatshield
sudo chown ubuntu:ubuntu /opt/threatshield
success "Directory /opt/threatshield created"

# ─────────────────────────────────────────────────────────────
# STEP 4: Python virtual environment
# ─────────────────────────────────────────────────────────────
info "STEP 4: Creating Python virtual environment..."
python3 -m venv /opt/threatshield/venv
source /opt/threatshield/venv/bin/activate
success "Virtual environment created at /opt/threatshield/venv"

# ─────────────────────────────────────────────────────────────
# STEP 5: Install Python packages
# ─────────────────────────────────────────────────────────────
info "STEP 5: Installing Python packages (this takes 2-3 minutes)..."
pip install --quiet --upgrade pip
pip install --quiet \
    xgboost \
    scikit-learn \
    pandas \
    numpy \
    scapy \
    joblib \
    flask \
    pyarrow
success "Python packages installed: xgboost, scapy, flask, pandas, numpy"

# ─────────────────────────────────────────────────────────────
# STEP 6: Copy uploaded files (user must have done scp first)
# ─────────────────────────────────────────────────────────────
info "STEP 6: Checking for model file..."
if [ -f "/home/ubuntu/idps_xgboost_model.pkl" ]; then
    cp /home/ubuntu/idps_xgboost_model.pkl /opt/threatshield/
    success "Model file copied to /opt/threatshield/"
else
    warn "⚠️  Model file not found at /home/ubuntu/idps_xgboost_model.pkl"
    warn "    Please SCP it manually:  scp idps_xgboost_model.pkl ubuntu@YOUR_EC2_IP:/home/ubuntu/"
fi

if [ -f "/home/ubuntu/idps_engine.py" ]; then
    cp /home/ubuntu/idps_engine.py /opt/threatshield/
    success "idps_engine.py copied"
fi

if [ -f "/home/ubuntu/dashboard.py" ]; then
    cp /home/ubuntu/dashboard.py /opt/threatshield/
    success "dashboard.py copied"
fi

# ─────────────────────────────────────────────────────────────
# STEP 7: Create empty events.json
# ─────────────────────────────────────────────────────────────
info "STEP 7: Creating initial log files..."
touch /opt/threatshield/idps.log
echo "[]" > /opt/threatshield/events.json
sudo chmod 666 /opt/threatshield/events.json /opt/threatshield/idps.log
success "Log files initialized"

# ─────────────────────────────────────────────────────────────
# STEP 8: systemd service for IDPS engine
# ─────────────────────────────────────────────────────────────
info "STEP 8: Creating systemd service for IDPS engine..."
sudo bash -c 'cat > /etc/systemd/system/idps.service << EOF
[Unit]
Description=THREATSHIELD AI-IDPS Engine
After=network.target
Wants=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/threatshield
ExecStart=/opt/threatshield/venv/bin/python3 /opt/threatshield/idps_engine.py
Restart=always
RestartSec=5
StandardOutput=append:/opt/threatshield/idps.log
StandardError=append:/opt/threatshield/idps.log
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
EOF'
success "idps.service created"

# ─────────────────────────────────────────────────────────────
# STEP 9: systemd service for Dashboard
# ─────────────────────────────────────────────────────────────
info "STEP 9: Creating systemd service for Dashboard..."
sudo bash -c 'cat > /etc/systemd/system/idps-dashboard.service << EOF
[Unit]
Description=THREATSHIELD Live Dashboard
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/opt/threatshield
ExecStart=/opt/threatshield/venv/bin/python3 /opt/threatshield/dashboard.py
Restart=always
RestartSec=5
StandardOutput=append:/opt/threatshield/dashboard.log
StandardError=append:/opt/threatshield/dashboard.log
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
EOF'
success "idps-dashboard.service created"

# ─────────────────────────────────────────────────────────────
# STEP 10: Enable & start services
# ─────────────────────────────────────────────────────────────
info "STEP 10: Enabling and starting services..."
sudo systemctl daemon-reload
sudo systemctl enable idps
sudo systemctl enable idps-dashboard
sudo systemctl start idps-dashboard

# Start IDPS only if model exists
if [ -f "/opt/threatshield/idps_xgboost_model.pkl" ]; then
    sudo systemctl start idps
    success "IDPS engine service started"
else
    warn "⚠️  IDPS engine NOT started — copy model file first, then run:"
    warn "    sudo systemctl start idps"
fi
success "Dashboard service started"

# ─────────────────────────────────────────────────────────────
# STEP 11: UFW Firewall rules
# ─────────────────────────────────────────────────────────────
info "STEP 11: Configuring firewall (UFW)..."
sudo ufw --force enable
sudo ufw allow ssh        # port 22 — keep SSH open!
sudo ufw allow 5000/tcp   # dashboard port
sudo ufw allow 80/tcp     # HTTP (optional)
success "Firewall configured: SSH(22) and Dashboard(5000) allowed"

# ─────────────────────────────────────────────────────────────
# DONE
# ─────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}======================================================${NC}"
echo -e "${GREEN}   ✅  THREATSHIELD SETUP COMPLETE!${NC}"
echo -e "${GREEN}======================================================${NC}"
echo ""

# Get public IP
PUBLIC_IP=$(curl -s http://checkip.amazonaws.com 2>/dev/null || echo "YOUR_EC2_PUBLIC_IP")

echo -e "${YELLOW}📊 DASHBOARD:${NC}      http://$PUBLIC_IP:5000"
echo -e "${YELLOW}📄 IDPS LOG:${NC}       /opt/threatshield/idps.log"
echo -e "${YELLOW}📄 EVENTS JSON:${NC}    /opt/threatshield/events.json"
echo ""
echo -e "${BLUE}Useful commands:${NC}"
echo "  sudo systemctl status idps             # check IDPS status"
echo "  sudo systemctl status idps-dashboard   # check dashboard status"
echo "  sudo systemctl restart idps            # restart IDPS"
echo "  tail -f /opt/threatshield/idps.log     # watch live log"
echo "  sudo iptables -L INPUT -n              # show blocked IPs"
echo ""
