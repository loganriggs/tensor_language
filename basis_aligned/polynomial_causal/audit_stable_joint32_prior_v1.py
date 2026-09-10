"""Post-fit comparison with the existing MLP17 pronoun readout, CPU only.

No fit, token-guided component selection, or native behavioral inference.
Only joint32 components meeting its frozen mutual-match rule are compared.
"""
import hashlib
import json
import time
from pathlib import Path
import torch
import tiktoken
from paired_panel_bootstrap_v1 import PairedPanelBootstrap

P = Path(__file__).resolve().parent
CK = Path('/workspace/.hf_home/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240/pytorch_model.bin')

def main():
    tic = time.perf_counter()
    torch.set_num_threads(2)
    result = json.loads((P/'UNSUPERVISED_JOINT32_V1_RESULT.json').read_text())
    artifact = P/'UNSUPERVISED_JOINT32_V1_PROGRAM.pt'
    assert hashlib.sha256(artifact.read_bytes()).hexdigest() == result['artifact_sha256']
    programs = torch.load(artifact, map_location='cpu', weights_only=True)
    ids = [i for i, yes in enumerate(result['stability']['matched']) if yes]
    assert ids == [0, 9]
    sd = torch.load(CK, map_location='cpu', mmap=True, weights_only=True)
    U = sd['lm_head.weight'].double()
    D, L, R = [sd[f'transformer.h.17.mlp.{k}.weight'].double() for k in ['Down', 'Left', 'Right']]
    enc = tiktoken.get_encoding('gpt2')
    pronouns = [' he', ' she', ' they', ' He', ' She', ' They']
    token_ids = [enc.encode(t)[0] for t in pronouns]
    assert all(enc.decode([i]) == s for i, s in zip(token_ids, pronouns))
    u = U[token_ids].mean(0); u /= u.norm()
    c = u@D
    raw = L.T@(c[:, None]*R); S = (raw+raw.T)/2
    lam, eig = torch.linalg.eigh(S); v = eig[:, 0]
    committee = ((L@v).abs()+(R@v).abs()).topk(64).indices

    # H_k v minus its projection on v. Under reflection about zero:
    # Delta B(x) = -4 (v^T x) T x, T = D [H_k v - v(v^T H_k v)].
    def reflection_readers(a, b):
        av, bv = a@v, b@v
        return .5*(bv[:, None]*a+av[:, None]*b)-(av*bv)[:, None]*v
    native_t = reflection_readers(L, R)
    T = D@native_t
    M = U.T@U
    def inner(a, b): return ((M@a)*b).sum()
    def compare(t):
        n = inner(T,T)
        return dict(relative_tensor_error=float((inner(T-t,T-t)/n).clamp_min(0).sqrt()),
                    projection_recovery=float(inner(T,t)/n),
                    relative_tensor_norm=float((inner(t,t)/n).clamp_min(0).sqrt()))
    reports = {}
    for name in ['native', 'random']:
        js = ids if name == 'native' else [result['stability']['native_partner'][i] for i in ids]
        p = programs[name]; a, b, w = p['a'][js], p['b'][js], p['w'][:, js]
        components = []
        for local, j in enumerate(js):
            reader_basis = torch.linalg.qr(torch.stack([a[local], b[local]], 1)).Q
            profile = p['token_coefficients'][:50257, j]
            components.append(dict(index=j, old_axis_abs_cosines=[float((a[local]@v).abs()),float((b[local]@v).abs())],
                old_axis_projection_into_reader_span=float((reader_basis.T@v).norm()),
                top_positive=[enc.decode([int(i)]) for i in profile.topk(12).indices],
                top_negative=[enc.decode([int(i)]) for i in (-profile).topk(12).indices]))
        reports[name] = dict(components=components, zero_centered_reflection=compare(w@reflection_readers(a,b)))

    qk = {}
    for version in [1,2]:
        q = json.loads((P/f'CORRELATIVE_JOINT_QK_SUBSPACES_V{version}_RESULT.json').read_text())
        qk[str(version)] = {}
        for i,(panel,r) in enumerate(q['reports'].items()):
            bs = PairedPanelBootstrap(8,9114200+i)
            qk[str(version)][panel] = {fam:dict(relative_error=m['relative_error'],
                error_interval=bs.relative_l2(m['error_squared'],m['reference_squared']),
                relative_effect=m['relative_effect'],
                effect_interval=bs.relative_l2(m['effect_squared'],m['reference_squared'])) for fam,m in r['families'].items()}
    # Independent direct matrix bridge for the reflection identity, one native product.
    l, r = L[0], R[0]; H = (l[:,None]*r+r[:,None]*l)/2
    J = torch.eye(len(v),dtype=v.dtype)-2*v[:,None]*v
    direct = J@H@J-H
    t = native_t[0]; formula = -2*(v[:,None]*t+t[:,None]*v)
    bridge = float((direct-formula).norm()/direct.norm())
    assert bridge < 1e-10
    out = dict(scope='Post-fit prior-art comparison. Reconstructs old weight-defined axis in FP64; zero-centered reflection is not the historical empirical-mean reflection and is not a behavior test.',
        old_axis_definition='Smallest eigenvector of normalized six-pronoun mean-unembedding quadratic at MLP17; reflect_gender.py.',
        smallest_eigenvalues=lam[:3].tolist(), axis_eigen_residual=float((S@v-lam[0]*v).norm()/S.norm()),
        stable_components=reports, native64_zero_centered_reflection=compare(D[:,committee]@native_t[committee]),
        reflection_identity_relative_error=bridge,
        historical_centering_correction='For eigenaxis coordinate s and center mu, reflecting s to 2mu-s changes the class quadratic by 4 lambda mu (mu-s). Exact invariance requires this term vanish; eigenvector status alone is insufficient.',
        qk_paired=qk, qk_scope='Same previously opened rows; intervals resample authored groups, not independent OOD confirmation.',
        native_forwards=0, cpu_seconds=time.perf_counter()-tic, source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    outpath = P/'STABLE_JOINT32_PRIOR_V1_AUDIT.json'
    with outpath.open('x') as f: json.dump(out,f,indent=2); f.write('\n')
    print(json.dumps(out,indent=2))

if __name__ == '__main__': main()
