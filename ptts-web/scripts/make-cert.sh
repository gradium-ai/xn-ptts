#!/bin/bash
# Self-signed cert covering this machine's current addresses.
#
# WebGPU needs a secure context. `localhost` counts as one; a LAN address over
# plain http does not, so a phone pointed at http://192.168.x.x gets
# `navigator.gpu === undefined` and the page cannot run at all. https with a
# self-signed cert is the cheapest way to get a secure context on the LAN --
# the phone shows a warning once, and you tap through.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p certs

IPS=$(ifconfig | grep "inet " | awk '{print $2}' | grep -v '^127\.')
SAN="DNS:localhost,IP:127.0.0.1"
for ip in $IPS; do SAN="$SAN,IP:$ip"; done
echo "cert covers: $SAN"

openssl req -x509 -newkey rsa:2048 -nodes -days 365 \
  -keyout certs/dev.key -out certs/dev.crt \
  -subj "/CN=ptts-web dev" -addext "subjectAltName=$SAN" 2>/dev/null

echo "wrote certs/dev.crt and certs/dev.key"
