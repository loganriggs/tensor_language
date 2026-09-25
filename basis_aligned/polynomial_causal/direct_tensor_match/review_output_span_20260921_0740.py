"""Bounded review: four output-form spans, not 256 atom spans. CPU only.
Prospective diagnostic: all four cosines >= .99 across all three alternates.
Positive invertible output mixing, coordinate null, independent dense Gram.
"""
from pathlib import Path
import hashlib, json, time
import torch

P = Path(__file__).resolve().parent
torch.set_num_threads(2)
torch.set_grad_enabled(False)
torch.set_default_dtype(torch.float64)

def gram(a, b):
    L, R, W = a
    A, B, V = b
    K = .5 * ((L.T @ A) * (R.T @ B) + (L.T @ B) * (R.T @ A))
    return W @ K @ V.T

def compare(a, b):
    G, H, C = gram(a,a), gram(b,b), gram(a,b)
    e, U = torch.linalg.eigh(G)
    f, V = torch.linalg.eigh(H)
    assert min(e.min(),f.min()) > 1e-12 * max(e.max(),f.max())
    cos = torch.linalg.svdvals((U/e.sqrt()).T @ C @ (V/f.sqrt())).clamp(0,1)
    E = G + H - C - C.T
    perturb = torch.linalg.eigvalsh((E+E.T)/2).max().clamp_min(0).sqrt()
    # (I-P_A) B = (I-P_A)(B-A), for full-column-rank B.
    bound = min(1., float(perturb / f.min().sqrt()))
    observed = float((1-cos.min().square()).clamp_min(0).sqrt())
    assert observed <= bound + 1e-8
    return dict(principal_cosines=cos.tolist(), max_sine=observed,
                perturbation_spectral=float(perturb), sigma_min_reference=float(e.min().sqrt()),
                sigma_min_candidate=float(f.min().sqrt()), elementary_bound=bound,
                relative_frobenius_difference=float(E.trace().clamp_min(0).sqrt()/G.trace().sqrt()))

def main():
    start=time.monotonic()
    files=['SHARED_PRODUCT_NATIVE_INPUTS_V1.pt','PROFILED_PARTIAL_GRAPH_PROGRAMS_V1.pt','PROFILED_PARTIAL_GRAPH_FIT_V1.json']
    hashes={n:hashlib.sha256((P/n).read_bytes()).hexdigest() for n in files}
    d=torch.load(P/files[0],weights_only=True,map_location='cpu')
    programs=torch.load(P/files[1],weights_only=True,map_location='cpu')
    winner=json.loads((P/files[2]).read_text())['winner']
    root=torch.linalg.inv(d['inverse_root'])
    def factors(p):
        x=p['shared_mixed']
        return root@x['left_reader'],root@x['right_reader'],x['product_weights'].T/d['scales'][:4,None]
    ref=factors(programs[winner])
    records={k:compare(ref,factors(p)) for k,p in programs.items() if k!=winner}
    g=torch.Generator().manual_seed(20260921)
    L,R,W=ref
    mixing=torch.eye(4)+.1*torch.randn(4,4,generator=g)
    positive=compare(ref,(L,R,mixing@W))
    perm=torch.randperm(L.shape[0],generator=g)
    signs=torch.where(torch.rand(L.shape[0],generator=g)>.5,1.,-1.)[:,None]
    null=compare(ref,(L[perm]*signs,R[perm]*signs,W))
    raw=torch.einsum('or,ir,jr->oij',W,L,R)
    dense=(raw+raw.transpose(-1,-2))/2
    flat=dense.flatten(1)
    replay=float((flat@flat.T-gram(ref,ref)).norm()/(flat@flat.T).norm())
    assert replay<1e-12 and min(positive['principal_cosines'])>1-1e-10
    unchanged=all(hashlib.sha256((P/n).read_bytes()).hexdigest()==h for n,h in hashes.items())
    assert unchanged
    out=dict(reference=winner,prospective_cosine_gate=.99,records=records,positive_output_mixing=positive,
             coordinate_null=null,dense_gram_relative_error=replay,source_sha256=hashes,
             sources_unchanged=unchanged,passes=all(min(r['principal_cosines'])>=.99 for r in records.values()),
             seconds=time.monotonic()-start,scope='Four covariance-weighted quadratic output forms; opened fitted weights, no native forwards or causal claim. Affine branches and private mode3 unchanged and outside this comparison.')
    target=P/'REVIEW_OUTPUT_SPAN_20260921_0740.json'
    with target.open('x') as f: json.dump(out,f,indent=2); f.write('\n')
    print(json.dumps(out,indent=2))

if __name__=='__main__': main()
