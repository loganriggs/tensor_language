# circuit_h126: WHAT IS HEAD 12.6? (S1505/07: it appears in 10 of 22 weights-only
# ensembles — a suspected punctuation/formatting generalist.) One removal arm
# (12.6 -> its optimal constant) vs clean, scored on ALL 22 class masks at once +
# global. Registered:
#   pred_a 12.6 alone raises class CE by >= .02 on >= 8 of the 22 classes.
#   pred_b its global rise <= .02 (broad but still punctuation-shaped, not generic).
#   pred_c classes where the screens CHOSE it are hit harder: mean rise (chosen) >=
#          2x mean rise (not chosen).
import json, time, sys, re, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl
import tiktoken

D = 1152; T = 256
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'circuit_h126_results.json'
NR = 960
H = m.transformer.h
ENC = tiktoken.get_encoding('gpt2')
CONSTS = torch.load(PT + 'opt_ablation_consts_all.pt', map_location='cpu')
C126 = CONSTS['head12.6'].to(DEV).float()
HOOK = {'on': False}


def h126_hook(mod, args):
    if HOOK['on']:
        x = args[0].clone()
        x[:, :, 6 * 128:7 * 128] = C126.to(x.dtype)
        return (x,)
    return None


def class_masks():
    V = 50257
    def rx(pat):
        v = torch.zeros(V, dtype=torch.bool)
        for t in range(V):
            if re.match(pat, ENC.decode([t])):
                v[t] = True
        return v
    C = {}
    nl = torch.zeros(V, dtype=torch.bool)
    for t in range(V):
        if '\n' in ENC.decode([t]):
            nl[t] = True
    C['newline'] = nl
    C['capitalized'] = rx(r'^ [A-Z]')
    C['digits'] = rx(r'^ ?[0-9]+$')
    C['open_quote'] = rx(r'^ ?["\u201c]$|^ "')
    C['close_paren'] = rx(r'^\)|^ ?\)$')
    C['comma'] = rx(r'^,$')
    C['question'] = rx(r'^\?$| \?$')
    C['the'] = rx(r'^ the$| The$|^The$')
    C['of'] = rx(r'^ of$')
    C['is'] = rx(r'^ is$| was$| are$')
    C['months'] = rx(r'^ (January|February|March|April|May|June|July|August|September|October|November|December)$')
    C['units'] = rx(r'^ ?(km|kg|mg|cm|mm|GB|MB|KB|Hz|MHz|GHz|mph|lbs|oz|ml)$')
    C['not'] = rx(r"^ not$|^ n\'t$|^n\'t$")
    C['close_quote'] = rx(r'^["\u201d]$|^ ?"$')
    C['colon'] = rx(r'^:$')
    C['semicolon'] = rx(r'^;$')
    C['to'] = rx(r'^ to$')
    C['and'] = rx(r'^ and$|^ or$|^ but$')
    C['years'] = rx(r'^ ?(19|20)[0-9]{2}$')
    C['said'] = rx(r'^ (said|says|told|asked|replied)$')
    C['ing'] = rx(r'^[a-z]+ing$| [a-z]+ing$')
    C['dollar'] = rx(r'^ ?[$\u00a3\u20ac]$')
    return C


CHOSEN = {'newline', 'capitalized', 'digits', 'open_quote', 'close_paren', 'comma',
          'question', 'the', 'close_quote', 'colon', 'semicolon', 'said'}


@torch.no_grad()
def fwd(idx):
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in H:
        x, v1 = blk(x, v1, x0)
    return 30.0 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30.0)


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    EVR = cl.fineweb_rows(NR, skip=7000)[:, :T + 1].contiguous()
    CLS = class_masks()
    hk = H[12].attn.c_proj.register_forward_pre_hook(h126_hook)

    def measure(on):
        HOOK['on'] = on
        gs = 0.0; gn = 0
        cs = {cn: 0.0 for cn in CLS}; cnn = {cn: 0 for cn in CLS}
        for i in range(0, NR, 8):
            bb = EVR[i:i + 8].to(DEV)
            idx = bb[:, :-1].contiguous(); tg = bb[:, 1:].contiguous()
            lo = fwd(idx).float()
            ce = F.cross_entropy(lo.reshape(-1, lo.shape[-1]), tg.reshape(-1),
                                 reduction='none').view(tg.shape)
            mk = torch.ones_like(tg, dtype=torch.bool); mk[:, :64] = False
            gs += float(ce[mk].sum()); gn += int(mk.sum())
            for cn, v in CLS.items():
                cm = v.to(DEV)[tg] & mk
                cs[cn] += float(ce[cm].sum()); cnn[cn] += int(cm.sum())
        HOOK['on'] = False
        return gs / max(gn, 1), {cn: cs[cn] / max(cnn[cn], 1) for cn in CLS}, cnn

    g0, c0, nn = measure(False)
    g1, c1, _ = measure(True)
    hk.remove()
    rises = {cn: round(c1[cn] - c0[cn], 4) for cn in CLS}
    gl = round(g1 - g0, 4)
    print('global rise', gl)
    for cn in sorted(rises, key=lambda c: -rises[c]):
        print(f"  {cn:12} {rises[cn]:+.4f} (n={nn[cn]})")

    hit = [cn for cn in CLS if rises[cn] >= 0.02]
    ch = [rises[cn] for cn in CLS if cn in CHOSEN]
    nc = [rises[cn] for cn in CLS if cn not in CHOSEN]
    mean_ch = sum(ch) / len(ch); mean_nc = sum(nc) / len(nc)
    pa = len(hit) >= 8
    pb = gl <= 0.02
    pc = mean_ch >= 2 * max(mean_nc, 1e-6)
    out = {'global_rise': gl, 'class_rises': rises, 'n_positions': nn,
           'n_hit_02': len(hit), 'mean_chosen': round(mean_ch, 4),
           'mean_not_chosen': round(mean_nc, 4),
           'pred_a_8_classes': bool(pa), 'pred_b_global_le_02': bool(pb),
           'pred_c_chosen_2x': bool(pc), 'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"pred_a {pa} | pred_b {pb} | pred_c {pc}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
