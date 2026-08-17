"""Edge universality by kinship: do the certified bilin18 edges replicate in the
sibling atlases? For each of bilin12, sqrd12, swiglu18: (i) relay
directionality (fraction of attention components whose top MLP kinship partner
is upstream/same-layer); (ii) the interchange analog (last attention ~
second-to-last MLP) in the relative form.

REGISTERED PREDICTIONS: (a) relay directionality >= 60% in all three siblings
(shuffled nulls ~50-60%); (b) the interchange analog certifies (>= median-other
+0.05) in at least the bilinear sibling (bilin12: attn11~mlp10); (c) exploratory
for the cross-family pair, reported without a bar."""
import json, torch, time
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'family_edge_kinship_results.json'

def spearman(a,b):
    ra=a.argsort().argsort().float(); rb=b.argsort().argsort().float()
    ra=(ra-ra.mean())/ra.std().clamp_min(1e-9)
    rb=(rb-rb.mean())/rb.std().clamp_min(1e-9)
    return float((ra*rb).mean())

def main():
    t0=time.time()
    atl={'bilin12':(PT+'bilin12_atlas.pt',12),
         'sqrd12':(PT+'sqrd12_atlas.pt',12),
         'swiglu18':(PT+'swiglu18_atlas.pt',18)}
    g=torch.Generator().manual_seed(0)
    out={}
    for name,(path,NL) in atl.items():
        fps={k:v.float() for k,v in torch.load(path)['fingerprints'].items()}
        up=0; upn=0
        for a in range(NL):
            fa=fps[f'attn{a}']
            best=max(range(NL),key=lambda mj:abs(spearman(fa,fps[f'mlp{mj}'])))
            if best<=a: up+=1
            fan=fa[torch.randperm(len(fa),generator=g)]
            bestn=max(range(NL),key=lambda mj:abs(spearman(fan,fps[f'mlp{mj}'])))
            if bestn<=a: upn+=1
        # interchange analog: last attn ~ second-to-last mlp
        la=NL-1; lm=NL-2
        ks={mj:abs(spearman(fps[f'attn{la}'],fps[f'mlp{mj}'])) for mj in range(NL)}
        others=[v for mj,v in ks.items() if mj!=lm]
        med=sorted(others)[len(others)//2]
        cert=(ks[lm]-med)>=0.05
        out[name]={'upstream_frac':up/NL,'null_frac':upn/NL,
                   'interchange_k':ks[lm],'median_other':med,
                   'certified':bool(cert)}
        print(f'{name:9s}: upstream {up}/{NL} (null {upn}/{NL}) | '
              f'attn{la}~mlp{lm} {ks[lm]:.3f} vs med {med:.3f} '
              f'{"CERT" if cert else "no"}',flush=True)
    pa=all(v['upstream_frac']>=0.6 for v in out.values())
    pb=out['bilin12']['certified']
    out['pred_a']=bool(pa); out['pred_b']=bool(pb)
    print(f"\n(a) relay directionality universal (>=60% all): "
          f"{'HELD' if pa else 'FAILED'}")
    print(f"(b) bilin12 interchange analog certifies: {'HELD' if pb else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
