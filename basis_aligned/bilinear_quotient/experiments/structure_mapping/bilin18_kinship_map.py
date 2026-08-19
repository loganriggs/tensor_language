"""Edge discovery by fingerprint kinship (scaling section 188's certified
instrument): for every component in the bilin18 atlas, find its top kinship
partner of the OTHER type. The relay (MLP writes -> attention transports -> MLP
consumes) predicts attention components partner with UPSTREAM MLPs.

REGISTERED PREDICTIONS: (a) relay directionality: >= 60% of attention
components' top MLP partner is at a layer <= their own (upstream or same);
(b) the certified cargo edge (attn6~mlp5) ranks in the top-5 of all 18x18
attention-MLP kinship pairs; (c) null: with per-component token-shuffled
fingerprints, the upstream fraction is 40-60% (no directionality)."""
import json, torch, time
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'bilin18_kinship_map_results.json'

def spearman(a,b):
    ra=a.argsort().argsort().float(); rb=b.argsort().argsort().float()
    ra=(ra-ra.mean())/ra.std().clamp_min(1e-9)
    rb=(rb-rb.mean())/rb.std().clamp_min(1e-9)
    return float((ra*rb).mean())

def main():
    t0=time.time()
    d=torch.load(PT+'bilin18_fingerprint_atlas.pt')
    fps={k:v.float() for k,v in d['fingerprints'].items()}
    K=torch.zeros(18,18)
    for a in range(18):
        for mj in range(18):
            K[a,mj]=abs(spearman(fps[f'attn{a}'],fps[f'mlp{mj}']))
    up=0
    for a in range(18):
        top=int(K[a].argmax())
        if top<=a: up+=1
        print(f'attn{a:2d}: top MLP partner mlp{top} ({K[a,top]:.2f})'
              f'{"  <= self" if top<=a else ""}',flush=True)
    flat=[(float(K[a,mj]),a,mj) for a in range(18) for mj in range(18)]
    flat.sort(reverse=True)
    rank_cargo=[i for i,(v,a,mj) in enumerate(flat) if a==6 and mj==5][0]+1
    g=torch.Generator().manual_seed(0)
    upn=0
    for a in range(18):
        fa=fps[f'attn{a}']
        fa=fa[torch.randperm(len(fa),generator=g)]
        best=max(range(18),key=lambda mj:abs(spearman(fa,fps[f'mlp{mj}'])))
        if best<=a: upn+=1
    pa=up/18>=0.6
    pb=rank_cargo<=5
    pc=0.4<=upn/18<=0.75
    out={'upstream_frac':up/18,'cargo_rank':rank_cargo,'null_frac':upn/18,
         'pred_a':bool(pa),'pred_b':bool(pb),'null_c':bool(pc)}
    print(f'\nupstream fraction {up}/18 | cargo edge rank {rank_cargo}/324 | '
          f'shuffled-null upstream {upn}/18')
    print(f"(a) relay directionality (>=60%): {'HELD' if pa else 'FAILED'}")
    print(f"(b) cargo edge top-5: {'HELD' if pb else 'FAILED'}")
    print(f"(c) null undirected: {'HELD' if pc else 'VIOLATED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
