#!/bin/bash
# Preserve the historical executor paths on a non-root workstation.
set -euo pipefail
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
state="${BILIN18_LOCAL_STATE:-/home/loganriggs/.local/share/bilin18}"
if [ "${BILIN18_PATH_BACKEND:-bwrap}" = proot ]; then
  exec "$state/proot/usr/bin/proot" \
    -b "$repo:/workspace/tensor_language" \
    -b "$repo/../theseus-bench:/workspace/theseus-bench" \
    -b "$state/hf_home:/workspace/.hf_home" \
    -b "$state/venv:/venv/main" -w /workspace/tensor_language \
    /usr/bin/env HF_HOME=/workspace/.hf_home \
    PATH="$state/bin:/venv/main/bin:/usr/local/bin:/usr/bin:/bin" \
    PYTHONPATH=/workspace/tensor_language:/workspace/tensor_language/basis_aligned/polynomial_causal:/workspace/tensor_language/basis_aligned/bilinear_quotient \
    "$@"
fi
exec /usr/bin/bwrap --tmpfs / \
  --ro-bind /usr /usr --symlink usr/bin /bin --symlink usr/sbin /sbin \
  --symlink usr/lib /lib --symlink usr/lib64 /lib64 \
  --ro-bind /etc /etc --dev-bind /dev /dev --proc /proc --ro-bind /sys /sys \
  --bind /home /home --bind /tmp /tmp --bind /run /run \
  --dir /workspace --bind "$repo" /workspace/tensor_language \
  --bind "$repo/../theseus-bench" /workspace/theseus-bench \
  --bind "$state/hf_home" /workspace/.hf_home \
  --dir /venv --bind "$state/venv" /venv/main \
  --setenv HF_HOME /workspace/.hf_home \
  --setenv PATH "$state/bin:/venv/main/bin:/usr/local/bin:/usr/bin:/bin" \
  --setenv PYTHONPATH "/workspace/tensor_language:/workspace/tensor_language/basis_aligned/polynomial_causal:/workspace/tensor_language/basis_aligned/bilinear_quotient" \
  --chdir /workspace/tensor_language "$@"
