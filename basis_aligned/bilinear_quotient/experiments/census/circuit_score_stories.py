"""CIRCUIT PIPELINE Stage E: mechanical semantic scoring. Each circuit
story from Stage D carries a membership rule (target_in / prev_in /
is_induction / pos_min / pos_max, AND-ed). Apply every rule to the HELD-OUT
(replication-half) well-predicted tokens; a story is SEMANTICALLY CERTIFIED
when on held-out data: precision >= max(5x the circuit's base rate, 0.15),
the rule fires on >= 10 tokens, and recall >= 0.15.

REGISTERED PREDICTIONS: (a) >= 40 of the storied circuits certify (first
mass pass); (b) >= 80% of non-null rules fire on >= 10 held-out tokens
(rules are executable, not vacuous); (c) median precision lift among
certified >= 5x base rate; (d) shuffled-null control: applying each
certified rule to a size-matched random token set yields precision at base
rate (median lift <= 1.5x)."""
import json, sys, glob, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
from bilin18_joint_removal import FW
import tiktoken
enc=tiktoken.get_encoding('gpt2')
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'circuit_score_stories_results.json'
R0=300

def main():
    t0=time.time()
    reg=torch.load(PT+'circuits_registry.pt',weights_only=False)
    rep=reg['rep_mask']; ar=reg['assign_rep']
    idx_rep=rep.nonzero().squeeze(1)
    feats=[]
    for gi in idx_rep.tolist():
        row=R0+gi//256; pos=gi%256
        t=int(FW[row,pos+1]); p=int(FW[row,pos])
        ind=t in FW[row,:pos+1].tolist()
        feats.append((enc.decode([t]),enc.decode([p]),ind,pos))
    stories=[]
    for f in sorted(glob.glob(PT+'circuits_stageD/stories_*.json')):
        stories+=json.load(open(f))
    def fires(rule,ft):
        tg,pv,ind,pos=ft
        if 'target_in' in rule and tg not in rule['target_in']: return False
        if 'prev_in' in rule and pv not in rule['prev_in']: return False
        if rule.get('is_induction') and not ind: return False
        if 'pos_min' in rule and pos<rule['pos_min']: return False
        if 'pos_max' in rule and pos>rule['pos_max']: return False
        return True
    N=len(feats)
    certified=[]; ex=0; nonnull=0; lifts=[]; nulllifts=[]
    g=torch.Generator().manual_seed(0)
    for s in stories:
        if s.get('rule') is None: continue
        nonnull+=1
        j=s['circuit_id']
        member=(ar==j)
        nb=int(member.sum())
        if nb==0: continue
        base_rate=nb/N
        hit=torch.tensor([fires(s['rule'],ft) for ft in feats])
        nf=int(hit.sum())
        if nf>=10: ex+=1
        else: continue
        prec=float((hit&member).sum())/nf
        rec=float((hit&member).sum())/nb
        lift=prec/max(base_rate,1e-9)
        rnd=torch.zeros(N,dtype=torch.bool)
        rnd[torch.randperm(N,generator=g)[:nb]]=True
        nprec=float((hit&rnd).sum())/nf
        nulllifts.append(nprec/max(base_rate,1e-9))
        ok=prec>=max(5*base_rate,0.15) and rec>=0.15
        if ok:
            lifts.append(lift)
            certified.append({'circuit_id':j,'story':s['story'],
                'precision':round(prec,3),'recall':round(rec,3),
                'lift':round(lift,1),'base_rate':round(base_rate,4),
                'fires':nf})
    lifts.sort(); nulllifts.sort()
    medlift=lifts[len(lifts)//2] if lifts else 0
    mednull=nulllifts[len(nulllifts)//2] if nulllifts else 0
    pa=len(certified)>=40
    pb=nonnull>0 and ex/nonnull>=0.80
    pc=medlift>=5
    pd=mednull<=1.5
    json.dump(certified,open(PT+'circuits_semantic_certified.json','w'),
              indent=1)
    out={'storied':nonnull,'executable':ex,'certified':len(certified),
         'median_lift':medlift,'median_null_lift':mednull,
         'pred_a':bool(pa),'pred_b':bool(pb),'pred_c':bool(pc),
         'pred_d':bool(pd)}
    print(f'stories {nonnull} | executable {ex} | CERTIFIED '
          f'{len(certified)} | median lift {medlift:.1f}x | null lift '
          f'{mednull:.1f}x')
    print(f"(a) >=40 certified: {'HELD' if pa else 'FAILED'}")
    print(f"(b) >=80% executable: {'HELD' if pb else 'FAILED'}")
    print(f"(c) median lift >=5x: {'HELD' if pc else 'FAILED'}")
    print(f"(d) null lift <=1.5x: {'HELD' if pd else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
