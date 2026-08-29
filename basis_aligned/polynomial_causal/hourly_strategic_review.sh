#!/bin/bash
set -u

THREAD_ID="01a04100-e4ff-7aa0-9caa-eef5085c4177"
PROMPT="HOURLY STRATEGIC REVIEW: Step back from the current experiment. The goal is to reverse engineer the entire bilin18 tensor network into a predictive, manipulable, simpler tensor program. Inspect the latest git history, AGENT_BOARD, result artifacts, and running jobs. Until 2026-08-29T12:00Z, also open basis_aligned/polynomial_causal/EIGHT_HOUR_ENTRYPOINT_PLAN_2026-08-29.md and advance its Family-F critical path plus E1.1-E4.3 evidence cells; do not count a plan or unrun runner as an outcome. Then: summarize what fraction of the model is actually explained and the largest remaining gaps; brainstorm candidate next actions that exploit tensor, polynomial, gauge, causal, and program structure; prune them by expected information gain, causal relevance, whole-model composability, falsifiability, GPU cost, and redundancy with completed work; rank the top five in priority order with reasons; and execute the highest-priority safe unblocked action. Do not stop at a status report and do not wait for user confirmation for safe actions already inside the project scope. If a GPU job is running, use the interval for useful CPU-side analysis, implementation, tests, or consolidation. Pause only for a genuine decision, new authority, destructive action, or external blocker; state the exact blocker and continue any independent work. Do not merely repeat the previous plan. Look specifically for missing interfaces, unexplained residual CE, OOD failures, interaction and composition failures, and opportunities for certified simplification. Preserve preregistration, failures, and agent coordination."

exec /opt/nvm/versions/node/v24.19.0/bin/codex queue \
    --thread "$THREAD_ID" \
    --message "$PROMPT"
