"""SOP PROGRAMS v2 -- rerun step-3 program search for the top-16
census leaves with the ENRICHED library (65 features: class labels
ported after 351's 2/16). Update records, register passing programs
as circ_ features, regenerate viewer.
REGISTERED: (a) >=8/16 pass (bacc>=0.75, null<=0.6);
(b) median null <=0.6; (c) >=half of passes cite a class_* predicate."""
import json, subprocess, time
import census_lib as cl
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'

def main():
    t0=time.time()
    eb=json.load(open(PT+'explainer_batch_results.json'))
    leaves=sorted(eb['leaves'],key=lambda r:-r['conc'])[:16]
    f=cl.surface_features()
    npass=0; nulls=[]; ncls=0
    for r in leaves:
        tag=r['tag']
        p=cl.leaf_program(tag,f)
        if not p.get('ok'): print(tag,'skip',p); continue
        ok=p['bacc']>=0.75 and p['null']<=0.6
        nulls.append(p['null'])
        if ok:
            npass+=1
            if 'class_' in str(p['program']): ncls+=1
            try: cl.register_feature('circ_'+tag.replace('.','_'),
                {'kind':'program','program':p['program'],
                 'provenance':'sop_programs2','cert':f"heldout {p['bacc']}"})
            except ValueError: pass
        cl.write_circuit(tag,{'story':{'blind_name':'',
            'program':p['program'],'program_bacc':p['bacc'],
            'program_null':p['null'],
            'mechanism_level':'surface' if ok else 'none'}})
        print(f"{tag}: bacc {p['bacc']} null {p['null']} "
              f"{'PASS' if ok else 'fail'} {p['program']}",flush=True)
    subprocess.run(['python',PT+'make_circuit_viewer.py'],check=True)
    mednull=sorted(nulls)[len(nulls)//2] if nulls else 1
    pa=npass>=8; pb=mednull<=0.6; pc=ncls>=max(1,npass//2)
    print(f'passes {npass}/16 | median null {mednull:.2f} | class-cited {ncls}')
    print(f"(a) >=8/16: {'HELD' if pa else 'FAILED'}")
    print(f"(b) null <=0.6: {'HELD' if pb else 'FAILED'}")
    print(f"(c) class cited >= half: {'HELD' if pc else 'FAILED'}")
    json.dump({'n_pass':npass,'median_null':round(mednull,3),
               'n_class_cited':ncls,'pred_a':bool(pa),'pred_b':bool(pb),
               'pred_c':bool(pc),'runtime_s':time.time()-t0},
              open(PT+'sop_programs2_results.json','w'),indent=1)

if __name__=='__main__': main()
