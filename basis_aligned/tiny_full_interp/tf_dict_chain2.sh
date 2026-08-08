#!/bin/bash
cd /workspace/tensor_language/basis_aligned/tiny_full_interp
source /venv/main/bin/activate
LOG=tf_dict_chain.log
while pgrep -f "bash ./tf_dict_chain.sh" > /dev/null; do sleep 30; done
echo "chain2 start $(date)" >> $LOG
python -u tf_dict_addendum.py --iters 6 --held_seq 256 > tf_dict_addendum.out 2>&1
echo "addendum done $(date)" >> $LOG
python -u tf_dict_atoms.py --n 256 --k 2 --iters 6 > tf_dict_atoms.out 2>&1
echo "atoms done $(date)" >> $LOG
