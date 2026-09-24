#!/usr/bin/env bash
# Commit and push small result files (predictions, logs, tables). Weights are git-ignored.
# Safe with several pods pushing to the same branch: never leaves a half-finished rebase behind,
# and on conflicts in shared files keeps this pod's version.
set -u
msg="${1:-results update}"
exec 9>/tmp/ptx_push.lock && flock 9   # serialise pushes (heartbeat + queue)
G="git -c user.name=ptx-runner -c user.email=ptx-runner@users.noreply.github.com"
git rebase --abort >/dev/null 2>&1; git merge --abort >/dev/null 2>&1; rm -rf .git/rebase-merge .git/rebase-apply
git add results splits 2>/dev/null
$G commit -q -m "results: $msg" >/dev/null 2>&1   # nothing new is fine: older local commits still get pushed
for i in 1 2 3 4 5; do
  if git fetch -q origin main; then
    if ! $G rebase -q -X theirs origin/main >/dev/null 2>&1; then
      git rebase --abort >/dev/null 2>&1; rm -rf .git/rebase-merge .git/rebase-apply
      $G merge -q -X ours --no-edit origin/main >/dev/null 2>&1 || { git merge --abort >/dev/null 2>&1; echo "push: could not reconcile with origin/main"; }
    fi
    git push -q origin HEAD:main && exit 0
  fi
  sleep $((5 + RANDOM % 20))
done
echo "push failed (will retry after next run)"
