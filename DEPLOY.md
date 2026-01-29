# ScoreboardCam Deploy (Pi 4)

This is a clean, repeatable install from fresh Pi OS Lite to a running
ScoreboardCam service.

## 1) Flash Pi OS (headless)
- Use Raspberry Pi Imager.
- OS: Raspberry Pi OS Lite (64-bit).
- Advanced options:
  - Hostname: scoreboardcam (or your choice)
  - Enable SSH
  - Set username/password
  - Configure Wi-Fi (SSID, password, country)
  - Locale/timezone
- Boot the Pi.

## 2) First login
```
ssh <user>@scoreboardcam.local
```
If mDNS is not available, use your router client list to find the IP.

## 3) Base system setup
```
sudo apt update
sudo apt full-upgrade -y
sudo apt install -y git python3 python3-venv python3-pip \
  libatlas-base-dev libopenjp2-7 libtiff5 libjpeg62-turbo libwebp7 libopenblas0
sudo reboot
```

## 4) Clone and install the app
```
cd /opt
sudo git clone https://github.com/trevdude83/scoreboardcam.git rocketsessions-scoreboardcam
sudo chown -R $USER:$USER /opt/rocketsessions-scoreboardcam

cd /opt/rocketsessions-scoreboardcam
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## 5) Configure app
- Copy your config:
```
cp config.example.yaml config.yaml
```
- Paste in your deviceId/deviceKey, server baseUrl, camera settings, etc.

- Copy templates (PNG) into:
```
/opt/rocketsessions-scoreboardcam/models/templates/
```

## 6) Set static IP
Check which network service is active:
```
systemctl is-active NetworkManager
```

### If NetworkManager is active (recommended on Bookworm)
```
nmcli con show
nmcli con mod "<CONNECTION_NAME>" ipv4.addresses 192.168.0.50/24
nmcli con mod "<CONNECTION_NAME>" ipv4.gateway 192.168.0.1
nmcli con mod "<CONNECTION_NAME>" ipv4.dns "192.168.0.1 1.1.1.1"
nmcli con mod "<CONNECTION_NAME>" ipv4.method manual
nmcli con up "<CONNECTION_NAME>"
```

### If NetworkManager is inactive (dhcpcd)
```
sudo nano /etc/dhcpcd.conf
```
Add to the end:
```
interface wlan0
static ip_address=192.168.0.50/24
static routers=192.168.0.1
static domain_name_servers=192.168.0.1 1.1.1.1
```
Then:
```
sudo systemctl restart dhcpcd
```

## 7) Quick foreground test
```
source /opt/rocketsessions-scoreboardcam/.venv/bin/activate
cd /opt/rocketsessions-scoreboardcam
python -m src.main run --config config.yaml
```
Preview:
```
http://<pi-ip>:5055/preview.html
```

## 8) Install systemd service
```
sudo cp /opt/rocketsessions-scoreboardcam/service/scoreboardcam.service \
  /etc/systemd/system/scoreboardcam.service
sudo systemctl daemon-reload
sudo systemctl enable --now scoreboardcam
sudo systemctl status scoreboardcam
```

## 9) Common checks
```
hostname -I
journalctl -u scoreboardcam -b --no-pager | tail -n 50
```

## 10) Update flow (when you pull new code)
```
cd /opt/rocketsessions-scoreboardcam
git pull
source .venv/bin/activate
pip install -r requirements.txt
sudo systemctl restart scoreboardcam
```
