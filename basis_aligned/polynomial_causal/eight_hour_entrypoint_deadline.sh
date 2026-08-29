#!/bin/bash
set -u

THREAD_ID="01a04100-e4ff-7aa0-9caa-eef5085c4177"
NOW="$(date -u +%Y-%m-%dT%H:%M)"
case "$NOW" in
    2026-08-29T11:5[5-9]|2026-08-29T12:[01][0-9]|2026-08-29T12:20) ;;
    *) exit 0 ;;
esac

PROMPT="EIGHT-HOUR DEADLINE AUDIT: Open basis_aligned/polynomial_causal/EIGHT_HOUR_ENTRYPOINT_PLAN_2026-08-29.md. Inspect current git, AGENT_BOARD, receipts, failures, running jobs, and every E1.1-E4.3 checkbox. Do not count plans or unrun code as outcomes. Finish any safe short unblocked experiment, then state exactly which cells have evidence, preserve failures, compare result-to-price and downstream causal utility, prune weak branches, rank the next one or two full experiments, update the plain-language explanation, commit, and push. Family F must have a numerical receipt or a precise preserved failure; never invent completion or silently change a preregistration."

exec /opt/nvm/versions/node/v24.19.0/bin/codex queue \
    --thread "$THREAD_ID" \
    --message "$PROMPT"
