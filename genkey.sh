#!/bin/bash
set -e
cd /home/linux
if [ ! -f ~/.ssh/id_ed25519 ]; then
  ssh-keygen -t ed25519 -C "linux@175.155.64.171 redstone_ai" -f ~/.ssh/id_ed25519 -N "" -q
  echo "key generated"
else
  echo "key already exists"
fi
echo "=== public key ==="
cat ~/.ssh/id_ed25519.pub
