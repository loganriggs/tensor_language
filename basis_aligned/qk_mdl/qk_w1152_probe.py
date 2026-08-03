"""Width-1152 slots-model interpretability probe (measurement block verbatim from
qk_v8_probe.py via qk_v9_common.full_probe, at D=1152 / slot size 48): causal
consumption graph, wiring agreement (all / effectual / top-10), dead blocks,
exact terms at blocks 4/6/9, token-determined + linear-in-embedding per layer.
Merges into qk_w1152.json under 'probe'."""
import os
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import qk_tokenline_train as Q
import qk_v9_common as C
from qk_w1152_train import patch_width, patch_data, WIDTH

if __name__ == '__main__':
    patch_width(WIDTH)
    patch_data()                     # Q.BATCH=8; held slice unchanged
    Q.gpu_guard(min_free=7000)
    model, ck = C.load_variant('qk_w1152_slots', 'W1152', None)
    C.full_probe(model, ck, save_cb=C.json_saver(f'{C.QK}/qk_w1152.json', 'probe'))
    print('saved qk_w1152.json (probe)', flush=True)
