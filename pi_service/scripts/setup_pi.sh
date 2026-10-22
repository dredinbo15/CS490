#!/usr/bin/env bash
# One-time Raspberry Pi setup. Run from anywhere:  bash pi_service/scripts/setup_pi.sh
# Installs Mosquitto, creates the Python environment, downloads YAMNet and
# installs the hub + Govee reader as services that start on boot.
set -euo pipefail

SERVICE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
RUN_USER="$(whoami)"
echo "Setting up in $SERVICE_DIR as user $RUN_USER"

echo "== Installing Mosquitto and Python tools"
sudo apt-get update
sudo apt-get install -y mosquitto mosquitto-clients python3-venv bluez

echo "== Letting the ESP32s connect to Mosquitto over your home network"
# Mosquitto 2.x only accepts local connections by default. This opens it to your LAN
# without a password, which is fine for the proof of concept on a home network.
# Add a password file before connecting anything outside your house.
sudo tee /etc/mosquitto/conf.d/hub.conf > /dev/null <<'EOF'
listener 1883 0.0.0.0
allow_anonymous true
EOF
sudo systemctl enable --now mosquitto
sudo systemctl restart mosquitto

echo "== Python environment"
python3 -m venv "$SERVICE_DIR/.venv"
"$SERVICE_DIR/.venv/bin/pip" install --upgrade pip
"$SERVICE_DIR/.venv/bin/pip" install -r "$SERVICE_DIR/requirements.txt"

echo "== YAMNet model"
if [ ! -f "$SERVICE_DIR/models/yamnet.tflite" ]; then
  bash "$SERVICE_DIR/scripts/get_models.sh"
fi

echo "== Unit tests"
(cd "$SERVICE_DIR" && .venv/bin/python -m unittest discover -s tests 2>&1 | tail -3)

echo "== Services (start on boot)"
for unit in hub govee; do
  sed -e "s|__DIR__|$SERVICE_DIR|g" -e "s|__USER__|$RUN_USER|g" \
    "$SERVICE_DIR/systemd/$unit.service" | sudo tee "/etc/systemd/system/home-$unit.service" > /dev/null
done
sudo systemctl daemon-reload
sudo systemctl enable home-hub home-govee

echo
echo "Done. Next:"
echo "  1. Find your Govees:     cd $SERVICE_DIR && .venv/bin/python -m hub.govee --discover"
echo "  2. Put their addresses in config.yaml (govee.sensors), and optionally your ntfy topic"
echo "  3. Start everything:     sudo systemctl start home-hub home-govee"
echo "  4. Watch the logs:       journalctl -u home-hub -u home-govee -f"
