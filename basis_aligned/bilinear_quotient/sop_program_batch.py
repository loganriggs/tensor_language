"""SOP PROGRAM BATCH -- 409: SOP step 3 (programs) over every
packed leaf (403's 25 + 408's 47). census_lib now supports the
diverse tree (use_state; grid parameterized). Programs are
doc-disjoint here in the strict sense: train/heldout split by
DOCUMENT id parity (curated_rows docid), not row parity (rows of
one document are adjacent in this corpus).
REGISTERED PREDICTIONS:
  (a) >=40% of packed leaves earn a program (heldout balanced
      acc >=0.75 with shuffled-label null <=0.6);
  (b) the null machinery is honest: median null across leaves
      <=0.6;
  (c) programs for passers written to sop_packs_programs.json
      with bacc, null, and the predicate list."""
import json, time, torch
import census_lib as cl
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'sop_program_batch_results.json'
PROGS=PT+'sop_packs_programs.json'

def main():
    t0=time.time()
    cl.use_state(PT+'census_state_diverse.pt')
    docid=torch.load(PT+'curated_rows.pt',map_location='cpu',
                     weights_only=False)['docid']
    tags=sorted(set(json.load(open(PT+'sop_packs_certified.json')))
                |set(json.load(open(PT+'sop_packs_shortlist.json'))))
    print(f'{len(tags)} packed leaves',flush=True)
    f=cl.surface_features()
    print(f'features built ({time.time()-t0:.0f}s)',flush=True)
    NF=cl.nflat()
    halfrow=(docid%2==0)
    half=halfrow[:,None].expand(-1,256).reshape(-1)
    bv=cl.base_ce()
    results=[]; progs={}
    for tag in tags:
        lf=cl.leaf(tag); mem=lf['member']
        g9=torch.Generator().manual_seed(3)
        memflat=torch.zeros(NF,dtype=torch.bool); memflat[mem]=True
        lo,hi=bv[mem].quantile(0.1),bv[mem].quantile(0.9)
        nonidx=torch.nonzero((~memflat)&(bv>=lo)&(bv<=hi)) \
            .squeeze(1)
        nonidx=nonidx[torch.randperm(len(nonidx),generator=g9)
                      [:len(mem)]]
        posA=mem[half[mem]]; posB=mem[~half[mem]]
        negA=nonidx[half[nonidx]]; negB=nonidx[~half[nonidx]]
        if min(len(posA),len(posB),len(negA),len(negB))<30:
            results.append({'tag':tag,'ok':False}); continue
        mask,prog=cl.rule_search(f,posA,negA)
        bacc=(float(mask[posB].float().mean())
              +1-float(mask[negB].float().mean()))/2
        lab=torch.cat([posA,negA])
        lab=lab[torch.randperm(len(lab),generator=g9)]
        mn,_=cl.rule_search(f,lab[:len(posA)],lab[len(posA):])
        null=(float(mn[posB].float().mean())
              +1-float(mn[negB].float().mean()))/2
        ok=bacc>=0.75 and null<=0.6
        results.append({'tag':tag,'ok':True,'bacc':round(bacc,3),
                        'null':round(null,3),
                        'pass':bool(ok),'program':prog})
        if ok:
            progs[tag]={'bacc':round(bacc,3),'null':round(null,3),
                        'program':prog,
                        'provenance':'sop_program_batch 409 '
                        '(doc-disjoint by docid parity)'}
        print(f"{tag}: bacc {bacc:.3f} null {null:.3f} "
              f"{'PASS' if ok else 'fail'} {prog}",flush=True)
    scored=[r for r in results if r.get('ok')]
    rate=sum(r['pass'] for r in scored)/max(len(scored),1)
    nulls=sorted(r['null'] for r in scored)
    mednull=nulls[len(nulls)//2] if nulls else 1.0
    json.dump(progs,open(PROGS,'w'),indent=1)
    pa=rate>=0.4; pb=mednull<=0.6; pc=len(progs)== \
        sum(r.get('pass') for r in scored)
    out={'results':results,'pass_rate':round(rate,3),
         'median_null':round(mednull,3),'n_programs':len(progs),
         'pred_a':bool(pa),'pred_b':bool(pb),'pred_c':bool(pc)}
    print(f"pass {rate:.2f} ({len(progs)}) | median null "
          f"{mednull:.3f}")
    for nm,v in (('a','>=40% earn a program'),
                 ('b','median null <=0.6'),
                 ('c','programs written for passers')):
        print(f"({nm}) {v}: "
              f"{'HELD' if out['pred_'+nm] else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
