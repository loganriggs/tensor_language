"""V9 interpretability probe (companion to qk_v9_train.py; measurement block
verbatim from qk_v8_probe.py via qk_v9_common.full_probe). Merges into
qk_v9.json under 'probe'."""
import os
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import qk_tokenline_train as Q
import qk_v9_common as C

if __name__ == '__main__':
    Q.gpu_guard(min_free=7000)
    model, ck = C.load_variant('qk_v9', 'V9', lambda li: C.window_vis(li, N=6))
    C.full_probe(model, ck, save_cb=C.json_saver(f'{C.QK}/qk_v9.json', 'probe'))
    print('saved qk_v9.json (probe)', flush=True)
