#!/bin/bash
# push_both.sh -- push both repos and CONFIRM BY RE-READING THE REMOTE.
#
# Written after LESSON 41's addendum: the inline helper this replaces ended in
# `git push ... 2>&1 | tail -1 && echo OK`, and a pipeline's exit status is the
# LAST command's, so it tested `tail` and printed OK on a failed push.
#
# The contract here is that success is never inferred from the push command.
# It is a positive re-read: fetch, then require `origin/main..HEAD` to be empty.
# Exits nonzero if EITHER repo still has unpushed commits, so a caller that
# ignores the message still sees the failure.
set -u
REPOS="/workspace/tensor_language /workspace/theseus-bench"
NAME=loganriggs
MAIL=logan.smith.5@gmail.com
rc=0
for r in $REPOS; do
    for _ in 1 2 3; do
        timeout 200 git -C "$r" fetch -q origin 2>/dev/null
        timeout 150 git -C "$r" -c user.name=$NAME -c user.email=$MAIL \
            merge -q --no-edit origin/main >/dev/null 2>&1
        timeout 200 git -C "$r" push -q origin main >/dev/null 2>&1
        timeout 200 git -C "$r" fetch -q origin 2>/dev/null
        n=$(git -C "$r" log --oneline origin/main..HEAD 2>/dev/null | wc -l)
        [ "$n" = 0 ] && break
        sleep 5
    done
    n=$(git -C "$r" log --oneline origin/main..HEAD 2>/dev/null | wc -l)
    if [ "$n" = 0 ]; then
        echo "PUSHED  $r  (verified: 0 commits ahead of origin/main)"
    else
        echo "FAILED  $r  ($n commits STILL UNPUSHED)"
        git -C "$r" log --oneline origin/main..HEAD | sed 's/^/          /'
        rc=1
    fi
done
exit $rc
