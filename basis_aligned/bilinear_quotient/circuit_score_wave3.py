"""Wave-3 scoring: cross-window by construction. Rules were written from
window-A examples + window-C EVEN-row examples; certification scores ONLY
window-C ODD rows, which no agent saw and no prior wave scored. Bars
unchanged: precision >= max(5x base, 0.15), recall >= 0.15, fires >= 10,
membership n >= 8. Passing entries join circuits_FINAL_certified.json
directly (this data is untouched -- no provisional stage needed).

REGISTERED PREDICTIONS: (a) >= 30 of ~141 wave-3 stories certify on the
unseen half-window (the topic-invariant protocol beats wave 1's 27 and
survives fresh data unlike wave 2's 33/39 failure); (b) class-level rules
(no token lists) certify at a higher rate than rules with token lists;
(c) median lift >= 5x."""
import json, sys, glob, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
from bilin18_joint_removal import FW
import tiktoken
enc=tiktoken.get_encoding('gpt2')
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'circuit_score_wave3_results.json'
DEV='cuda'

def tclass(tok):
    if '\n' in tok: return 'newline'
    s=tok.strip()
    if not s: return 'other'
    if s.isdigit(): return 'digit'
    if all(not c.isalnum() for c in s): return 'punct'
    if s[0].isupper(): return 'upper_start'
    if tok.startswith(' ') and s.isalpha(): return 'space_word'
    return 'other'

def main():
    t0=time.time()
    reg=torch.load(PT+'circuits_registry.pt',weights_only=False)
    big=torch.load(PT+'circuit_atlas_big.pt',weights_only=False)
    thr=torch.load(PT+'circuit_atlas_third.pt',weights_only=False)
    keys=reg['keys']
    Mb=torch.stack([big['fingerprints'][k].float().to(DEV) for k in keys])
    disc=reg['disc_mask'].to(DEV)
    mu=Mb[:,disc].mean(1,keepdim=True)
    sd=Mb[:,disc].std(1,keepdim=True).clamp_min(1e-8)
    M3=torch.stack([thr['fingerprints'][k].float().to(DEV) for k in keys])
    base3=thr['base'].float().to(DEV)
    wp=base3<=base3.median()
    Z3=((M3-mu)/sd)[:,wp].T
    Z3=Z3/Z3.norm(dim=1,keepdim=True).clamp_min(1e-8)
    C=reg['centroids'].to(DEV)
    a3=(Z3@C.T).argmax(1).cpu()
    idx3=wp.nonzero().squeeze(1).cpu()
    odd=((idx3//256)%2==1)          # odd rows of window C: unseen
    a3o=a3[odd]; idx3o=idx3[odd]
    R0=120
    feats=[]
    for gi in idx3o.tolist():
        row=R0+gi//256; pos=gi%256
        t=int(FW[row,pos+1]); p=int(FW[row,pos])
        p2=int(FW[row,pos-1]) if pos>=1 else -1
        tg=enc.decode([t])
        feats.append({'target':tg,'prev':enc.decode([p]),
            'prev2':enc.decode([p2]) if p2>=0 else '',
            'is_induction':t in FW[row,:pos+1].tolist(),
            'target_equals_prev':t==p,'target_class':tclass(tg),'pos':pos})
    def fires(rule,f):
        if 'target_in' in rule and f['target'] not in rule['target_in']: return False
        if 'prev_in' in rule and f['prev'] not in rule['prev_in']: return False
        if 'prev2_in' in rule and f['prev2'] not in rule['prev2_in']: return False
        if rule.get('is_induction') and not f['is_induction']: return False
        if rule.get('target_equals_prev') and not f['target_equals_prev']: return False
        if 'target_class_in' in rule and f['target_class'] not in rule['target_class_in']: return False
        if 'pos_min' in rule and f['pos']<rule['pos_min']: return False
        if 'pos_max' in rule and f['pos']>rule['pos_max']: return False
        return True
    N=len(feats)
    stories=[]
    for f in sorted(glob.glob(PT+'circuits_wave3/stories_*.json')):
        stories+=json.load(open(f))
    passed=[]; listy=[0,0]; classy=[0,0]; lifts=[]
    for s in stories:
        rule=s.get('rule')
        if rule is None: continue
        haslist=any(k in rule for k in ('target_in','prev_in','prev2_in'))
        j=s['circuit_id']
        member=(a3o==j); nb=int(member.sum())
        if nb<8: continue
        (listy if haslist else classy)[1]+=1
        hit=torch.tensor([fires(rule,f) for f in feats])
        nf=int(hit.sum())
        if nf<10: continue
        prec=float((hit&member).sum())/nf
        rec=float((hit&member).sum())/nb
        br=nb/N
        if prec>=max(5*br,0.15) and rec>=0.15:
            (listy if haslist else classy)[0]+=1
            lifts.append(prec/max(br,1e-9))
            passed.append({'circuit_id':j,'story':s['story'],
                'status':'wave3','precision':round(prec,3),
                'recall':round(rec,3),'lift':round(prec/br,1)})
    lifts.sort()
    medl=lifts[len(lifts)//2] if lifts else 0
    final=json.load(open(PT+'circuits_FINAL_certified.json'))
    known={c['circuit_id'] for c in final}
    final+= [p for p in passed if p['circuit_id'] not in known]
    json.dump(final,open(PT+'circuits_FINAL_certified.json','w'),indent=1)
    pa=len(passed)>=30
    rl=listy[0]/max(listy[1],1); rc=classy[0]/max(classy[1],1)
    pb=rc>=rl
    pc=medl>=5
    out={'wave3_pass':len(passed),'class_rate':f'{classy[0]}/{classy[1]}',
         'list_rate':f'{listy[0]}/{listy[1]}','median_lift':medl,
         'FINAL_total':len(final),
         'pred_a':bool(pa),'pred_b':bool(pb),'pred_c':bool(pc)}
    print(f'wave-3 pass {len(passed)} | class-only {classy[0]}/{classy[1]} '
          f'vs with-lists {listy[0]}/{listy[1]} | median lift {medl:.1f}x | '
          f'FINAL total {len(final)}')
    print(f"(a) >=30 pass: {'HELD' if pa else 'FAILED'}")
    print(f"(b) class-rules certify at >= list-rule rate: "
          f"{'HELD' if pb else 'FAILED'}")
    print(f"(c) median lift >=5x: {'HELD' if pc else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
