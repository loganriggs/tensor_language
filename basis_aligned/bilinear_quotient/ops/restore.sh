#!/bin/bash
# restore.sh -- rebuild this box's runtime after a Vast RECYCLE/DESTROY.
#
# ${WORKSPACE} is NOT volume-backed on this instance (verified:
# `vast-capabilities | jq '.instance.workspace_is_volume'` -> false), so a recycle
# wipes EVERYTHING except what is in git.  Written 2026-08-24 after exactly that
# happened mid-session: the repo survived (all work was pushed), but the venv,
# the HF cache, /workspace/rspd and the bqrunner service were all gone.
#
# Usage (from anywhere, as root):
#   git clone https://github.com/loganriggs/tensor_language.git /workspace/tensor_language
#   bash /workspace/tensor_language/basis_aligned/bilinear_quotient/ops/restore.sh
#
# Takes ~5 minutes.  Idempotent -- safe to re-run.
set -u
BQ=/workspace/tensor_language/basis_aligned/bilinear_quotient
say () { echo "=== $* ==="; }

say "1/6 python env (RTX 5090 = sm_120, needs a CUDA >= 12.8 wheel)"
source /venv/main/bin/activate
uv pip install torch --index-url https://download.pytorch.org/whl/cu128
uv pip install numpy scipy scikit-learn matplotlib tiktoken datasets transformers einops pandas
python - <<'PY'
import torch
assert torch.cuda.is_available(), 'no CUDA'
x = torch.randn(512, 512, device='cuda'); float((x @ x).sum())
print('torch', torch.__version__, torch.cuda.get_device_name(0), 'ok')
PY

say "2/6 rspd (Eckart-Young functional-core toolkit; imported by rspd_*.py)"
[ -d /workspace/rspd ] || git clone --depth 1 https://github.com/ThatE10/rspd.git /workspace/rspd

say "3/6 model weights -> \$HF_HOME (${HF_HOME:-unset})"
python - <<'PY'
from huggingface_hub import hf_hub_download
REPOS = ['Elriggs/gpt2-bilinear-sqrd-attn-18l-9h-1152embd',   # bilin18 -- the target model
         'Elriggs/gpt2-bilinear-sqrd-attn-12l-6h-768embd',    # bilin12
         'Elriggs/gpt2-bilinear-12l-6h-768embd',              # bilinsm12
         'Elriggs/gpt2-bilinear-swiglu-18l-9h-1152embd',      # swiglu18
         'Elriggs/gpt2-sqrd-attn-12l-6h-768embd']             # sqrd12
for r in REPOS:
    for f in ('config.json', 'pytorch_model.bin'):
        try:
            hf_hub_download(r, f); print('ok', r, f, flush=True)
        except Exception as e:
            print('FAIL', r, f, type(e).__name__, e, flush=True)
PY

say "4/6 FineWeb stream + gpt2 tokenizer warm-up"
python - <<'PY'
import sys; sys.path.insert(0, '/workspace/tensor_language/basis_aligned/bilinear_quotient')
import census_lib as cl
cl.use_state(cl.PT + 'census_state_diverse.pt')
r = cl.fineweb_rows(8)
print('fineweb rows', tuple(r.shape))
print(repr(cl.enc().decode(r[0, :40].tolist())))
PY

say "5/6 bqrunner service (drains queue.txt; canary on idle)"
cp "$BQ/ops/bqrunner.sh"   /opt/supervisor-scripts/bqrunner.sh
cp "$BQ/ops/bqrunner.conf" /etc/supervisor/conf.d/bqrunner.conf
chmod +x /opt/supervisor-scripts/bqrunner.sh
cp "$BQ/ops/bqrunner2.sh"   /opt/supervisor-scripts/bqrunner2.sh
cp "$BQ/ops/bqrunner2.conf" /etc/supervisor/conf.d/bqrunner2.conf
chmod +x /opt/supervisor-scripts/bqrunner2.sh
touch "$BQ/queue2.txt"
supervisorctl reread && supervisorctl update
sleep 3; supervisorctl status bqrunner

say "6/6 canary (the gate: model + atlases + data path must reproduce)"
cd "$BQ" && python bilin18_canary2.py 2>&1 | tail -4

cat <<'EOF'

STILL SESSION-LOCAL, recreate by hand:
  * the wake cron (CronCreate) -- it does NOT survive a session, let alone a recycle
  * the driver's memory dir /root/.claude/projects/-workspace-tensor-language/memory/
See SWARM_RUNBOOK.md §0 for the new-session bootstrap.
EOF
