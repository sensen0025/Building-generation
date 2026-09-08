#!/bin/bash
set -e
cd /home/linux/redstone_ai
# place .gitignore
cp /home/linux/redstone_ai/genkey.sh /dev/null 2>/dev/null || true
# init repo
if [ ! -d .git ]; then
  git init -b main 2>&1 | head -1
fi
git config user.name "linux"
git config user.email "linux@redstone.local"
# stage
git add -A
echo "=== staged file count ==="
git diff --cached --name-only | wc -l
echo "=== staged total size ==="
git diff --cached --name-only -z | du -ch --files0-from=- 2>/dev/null | tail -1
echo "=== top dirs by staged files ==="
git diff --cached --name-only | cut -d/ -f1 | sort | uniq -c | sort -rn | head -15
