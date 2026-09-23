#!/usr/bin/env bash
# Commit and push small result files (predictions, logs, tables). Weights are git-ignored.
set -u
msg="${1:-results update}"
exec 9>/tmp/ptx_push.lock && flock 9   # serialise pushes (heartbeat + queue)
git add results splits 2>/dev/null
git -c user.name="ptx-runner" -c user.email="ptx-runner@users.noreply.github.com" commit -q -m "results: $msg" || exit 0
for i in 1 2 3; do
  git pull -q --rebase origin main && git push -q origin HEAD:main && exit 0
  sleep 5
done
echo "push failed (will retry after next run)"
