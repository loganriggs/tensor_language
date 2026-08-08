#!/bin/bash
# Sequential chain, NO pgrep gating (the pgrep gate self-matched an agent
# command line and deadlocked; the brief warns about exactly this).
cd /workspace/tensor_language/basis_aligned/tiny_full_interp
source /venv/main/bin/activate
LOG=tf_dict_chain.log
echo "chain3 start $(date)" >> $LOG
python -u tf_dict_emb_run.py --iters 5 --held_seq 256 --est_seq 128 > tf_dict_emb.out 2>&1
echo "emb s0 done rc=$? $(date)" >> $LOG
python -u tf_dict_addendum.py --iters 6 --held_seq 256 > tf_dict_addendum.out 2>&1
echo "addendum done rc=$? $(date)" >> $LOG
python -u tf_dict_atoms.py --n 256 --k 2 --iters 6 > tf_dict_atoms.out 2>&1
echo "atoms done rc=$? $(date)" >> $LOG
python -u tf_dict_fold_run.py --stem tf_vanilla_d1_w128_b8192_s1 --quick --iters 6 --held_seq 256 --est_seq 128 > tf_dict_fold_s1.out 2>&1
echo "fold s1 done rc=$? $(date)" >> $LOG
python -u tf_dict_emb_run.py --stem tf_vanilla_d1_w128_b8192_s1 --quick --iters 5 --held_seq 256 --est_seq 128 > tf_dict_emb_s1.out 2>&1
echo "emb s1 done rc=$? $(date)" >> $LOG
echo "chain3 done $(date)" >> $LOG
