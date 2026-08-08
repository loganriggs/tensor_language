#!/bin/bash
# Detached chain for the ported-dictionary work.  Gated by EXACT-NAME pgrep
# (substring pgrep self-matches have killed runs in this program twice).
cd /workspace/tensor_language/basis_aligned/tiny_full_interp
source /venv/main/bin/activate
LOG=tf_dict_chain.log
echo "chain start $(date)" >> $LOG

wait_for () {   # $1 = exact script name
  while pgrep -f "python -u $1" > /dev/null; do sleep 20; done
}

wait_for tf_dict_fold_run.py
echo "fold run finished $(date)" >> $LOG

python -u tf_dict_emb_run.py --iters 5 --held_seq 256 --est_seq 128 \
    > tf_dict_emb.out 2>&1
echo "emb run finished $(date)" >> $LOG

# seed-1 confirmation (reduced sweeps, same controls) -- structure claims in
# this program need more than one seed.
python -u tf_dict_fold_run.py --stem tf_vanilla_d1_w128_b8192_s1 --quick \
    --iters 6 --held_seq 256 --est_seq 128 > tf_dict_fold_s1.out 2>&1
echo "fold s1 finished $(date)" >> $LOG
python -u tf_dict_emb_run.py --stem tf_vanilla_d1_w128_b8192_s1 --quick \
    --iters 5 --held_seq 256 --est_seq 128 > tf_dict_emb_s1.out 2>&1
echo "emb s1 finished $(date)" >> $LOG
echo "chain done $(date)" >> $LOG
