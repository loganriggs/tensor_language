"""Width-768 slots-model interpretability probe (measurement block verbatim from
qk_v8_probe.py via qk_v9_common.full_probe, at D=768 / slot size 32). Merges into
qk_w768.json under 'probe'."""
import os
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import qk_tokenline_train as Q
import qk_v9_common as C
from qk_w768_train import patch_width, WIDTH

if __name__ == '__main__':
    patch_width(WIDTH)
    Q.gpu_guard(min_free=7000)
    model, ck = C.load_variant('qk_w768_slots', 'W768', None)
    C.full_probe(model, ck, save_cb=C.json_saver(f'{C.QK}/qk_w768.json', 'probe'))
    print('saved qk_w768.json (probe)', flush=True)
