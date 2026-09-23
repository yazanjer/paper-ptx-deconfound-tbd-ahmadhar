#!/usr/bin/env bash
# Records the SHA-256 of PREREGISTRATION.md. External evaluation refuses to run if it changes later.
set -e
mkdir -p results
h=$(sha256sum PREREGISTRATION.md | cut -d' ' -f1)
if [ -f results/prereg_hash.txt ] && [ "$(cat results/prereg_hash.txt)" != "$h" ]; then
  echo "PREREGISTRATION.md differs from the frozen version"; exit 1
fi
echo "$h" > results/prereg_hash.txt
echo "frozen: $h"
