// Copy this file to secrets.h (same folder) and fill in your details.
// secrets.h is in .gitignore, so your WiFi password never ends up on GitHub.
#pragma once

#define WIFI_SSID     "your-wifi-name"      // must be a 2.4 GHz network; the ESP32 can't use 5 GHz
#define WIFI_PASSWORD "your-wifi-password"

// Where Mosquitto runs: the Pi's hostname with ".local", or its IP address.
// Examples: "security-hub.local" or "192.168.1.50"
// For testing before the Pi is ready, you can use your laptop's IP if Mosquitto runs there.
#define MQTT_HOST     "security-hub.local"
