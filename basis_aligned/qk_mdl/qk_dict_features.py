"""FEATURE INSPECTION (Logan 2026-07-22): are the layer-0 QK dictionary atoms meaningful?

Loads the saved seed-0 dictionary (qk_dict_l0_seed0.pt, n=1024 k=8 per head-branch), assigns every
vocabulary token its top-8 atoms via the trained encoder, and for a sample of head-branches dumps:
for the 8 most-used atoms and 8 seeded-random atoms (>= 20 users), the top-14 tokens by
absolute coefficient. Token strings are GPT-2 pieces (G-dot = leading space).

Writes qk_dict_features.md (+ .json); prints two branches as a preview.
"""
import json
import sys
import torch
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs
from tier2_folding import branch_factors
from transformers import AutoTokenizer

torch.manual_seed(0)
DEV = 'cuda'
QK = '/workspace/tensor_language/basis_aligned/qk_mdl'
BRANCHES = (('q1', 'k1'), ('q2', 'k2'))
SAMPLE_BI = (0, 1, 8, 9, 14, 17)          # (head, branch): h0b1 h0b2 h4b1 h4b2 h7b1 h8b2
K = 8

m, cfg = load_elriggs('bilin18')
NH, HD = cfg['n_head'], cfg['n_embd'] // cfg['n_head']
V = cfg['vocab_size']
tok = AutoTokenizer.from_pretrained('gpt2')

TAB = {}
for br, (qn, kn) in enumerate(BRANCHES, start=1):
    qh, kh = branch_factors(m, br)
    TAB[qn], TAB[kn] = qh.float().to(DEV), kh.float().to(DEV)
HB = [(h, qn, kn) for h in range(NH) for (qn, kn) in BRANCHES]

blob = torch.load(f'{QK}/qk_dict_l0_seed0.pt', map_location=DEV)

out_md = ['# Layer-0 QK dictionary atoms — token samples (seed-0 dict, n=1024, k=8)\n',
          'Per atom: top tokens by |coefficient| among tokens whose top-8 support includes the atom.',
          'Signs shown; GPT-2 pieces (Ġ = leading space).\n']
out_js = {}
g = torch.Generator().manual_seed(0)

for bi in SAMPLE_BI:
    h, qn, kn = HB[bi]
    X = torch.cat([TAB[qn][:, h], TAB[kn][:, h]], 1)
    Dn, b, We = blob[f'Dn{bi}'], blob[f'b{bi}'], blob[f'We{bi}']
    z = (X - b) @ We.T
    vals, idx = z.abs().topk(K, dim=1)
    coeff = torch.gather(z, 1, idx)                       # (V, K) signed
    usage = torch.zeros(Dn.shape[0], device=DEV)
    usage.index_add_(0, idx.reshape(-1), torch.ones(idx.numel(), device=DEV))
    eligible = (usage >= 20).nonzero().squeeze(1)
    top_atoms = usage.topk(8).indices.tolist()
    rand_atoms = eligible[torch.randperm(len(eligible), generator=g)[:8]].tolist()

    name = f'head {h}, branch {1 if qn == "q1" else 2}'
    out_md.append(f'\n## {name}  (head-branch {bi}; atoms used by >=1 token: '
                  f'{int((usage > 0).sum())}/1024)\n')
    out_js[bi] = {'name': name, 'atoms': {}}
    for label, atoms in (('most-used', top_atoms), ('random', rand_atoms)):
        for a in atoms:
            users = (idx == a).any(1).nonzero().squeeze(1)
            c_a = torch.where(idx[users] == a, coeff[users], torch.zeros_like(coeff[users])).sum(1)
            order = c_a.abs().argsort(descending=True)[:14]
            toks = [(tok.convert_ids_to_tokens(int(users[o])), round(float(c_a[o]), 2))
                    for o in order]
            out_js[bi]['atoms'][int(a)] = {'label': label, 'n_users': int(len(users)), 'top': toks}
            tokstr = '  '.join(f'`{t}`({c:+.1f})' for t, c in toks)
            out_md.append(f'- **atom {a}** ({label}, {len(users)} tokens): {tokstr}')

open(f'{QK}/qk_dict_features.md', 'w').write('\n'.join(out_md))
json.dump(out_js, open(f'{QK}/qk_dict_features.json', 'w'), indent=2)
print('\n'.join(out_md[:40]))
print(f'\nwrote qk_dict_features.md / .json')
