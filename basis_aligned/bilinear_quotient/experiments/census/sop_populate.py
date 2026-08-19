"""SOP POPULATE -- run SOP steps 1-3+6 for the 50 explainer_batch
leaves using already-measured causal data; leaf_program (step 3, ~2
min CPU each) for the top-16 by concentration; passing programs
registered as circ_<tag> features (compounding). Regenerates
circuits.html. Merges 349's bundle/tension structure into r.0.0.1
and stamps tension back-edges on partner records.
REGISTERED: (a) >=45 records written; (b) of the 16 programmed,
>=8 pass (bacc>=0.75, null<=0.6); (c) registry/viewer consistent
(no orphan files)."""
import json, subprocess, time
import census_lib as cl
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'

def main():
    t0=time.time()
    eb=json.load(open(PT+'explainer_batch_results.json'))
    bs=json.load(open(PT+'bundle_split_results.json'))
    leaves=sorted(eb['leaves'],key=lambda r:-r['conc'])
    tens_by_partner={e['tag']:e['value'] for e in bs['tension_edges']}
    npass=0; nrec=0
    for i,r in enumerate(leaves):
        tag=r['tag']
        upd={'components':r['probes'],
             'members':{'n':r['n_members'],
                        'indices':[int(x) for x in cl.leaf(tag)['member']]},
             'base_ce':{'member_mean':r['base_ce_member_mean'],
                        'frac_lt3':r['base_ce_frac_lt3']},
             'causal':{'abs_dce_members':r['abs_dce_members'],
                       'abs_dce_offslice':r['abs_dce_offslice'],
                       'concentration':r['conc'],
                       'dce_pos':r['dce_pos'],'dce_neg':r['dce_neg'],
                       'n_pos':r['n_pos'],'n_neg':r['n_neg'],
                       'minority_share':r['min_sign_share']},
             'certification':[{'test':'|dCE| concentration >=3x corpus',
                               'value':r['conc'],
                               'verdict':'HELD' if r['conc']>=3 else 'FAILED',
                               'source':'explainer_batch','date':'2026-08-19'}],
             'examples':{'top':r['top_examples'],
                         'random':r['random_examples'],
                         'rule':'top-3 by |score| + 3 seed-0 random'},
             'provenance':{'scripts':['explainer_batch.py','sop_populate.py'],
                           'sections':['348']}}
        if tag in tens_by_partner:
            upd['relations']={'tension':[{'tag':'r.0.0.1',
                'value':tens_by_partner[tag],
                'evidence':'members improve when r.0.0.1 machinery ablated'}]}
        if i<16 and r['conc']>=3:
            p=cl.leaf_program(tag)
            if p.get('ok'):
                ok=p['bacc']>=0.75 and p['null']<=0.6
                upd['story']={'blind_name':'','program':p['program'],
                              'program_bacc':p['bacc'],
                              'program_null':p['null'],
                              'mechanism_level':'surface' if ok else 'none'}
                if ok:
                    npass+=1
                    try: cl.register_feature('circ_'+tag.replace('.','_'),
                        {'kind':'program','program':p['program'],
                         'provenance':'sop_populate step3',
                         'cert':f"heldout {p['bacc']}"})
                    except ValueError: pass
                print(f'{tag}: bacc {p["bacc"]} null {p["null"]} '
                      f'{"PASS" if ok else "fail"}',flush=True)
        cl.write_circuit(tag,upd)
        nrec+=1
    # merge 349 into r.0.0.1
    cl.write_circuit('r.0.0.1',{
        'causal_bundles':{'wing_pos':bs['wing_pos_by_bundle'],
                          'wing_neg':bs['wing_neg_by_bundle'],
                          'pairwise_corr':bs['pairwise_corr_members'],
                          'structure':'push (m0 b0) minus brake (m3 b1/b2)',
                          'abbey_attribution':bs['abbey']},
        'relations':{'parent':'r.0.0',
                     'tension':bs['tension_edges']}})
    reg=json.load(open(PT+'circuits/registry.json'))
    import os
    files={m9['file'] for m9 in reg['circuits'].values()}
    orph=[f for f in os.listdir(PT+'circuits')
          if f.endswith('.json') and f!='registry.json' and f not in files]
    subprocess.run(['python',PT+'make_circuit_viewer.py'],check=True)
    pa=nrec>=45; pb=npass>=8; pc=len(orph)==0
    print(f'records {nrec} | programmed passes {npass}/16 | orphans {orph}')
    print(f"(a) >=45 records: {'HELD' if pa else 'FAILED'}")
    print(f"(b) >=8/16 program passes: {'HELD' if pb else 'FAILED'}")
    print(f"(c) no orphans: {'HELD' if pc else 'FAILED'}")
    json.dump({'n_records':nrec,'n_program_pass':npass,
               'pred_a':bool(pa),'pred_b':bool(pb),'pred_c':bool(pc),
               'runtime_s':time.time()-t0},
              open(PT+'sop_populate_results.json','w'),indent=1)
    print('done',time.time()-t0)

if __name__=='__main__': main()
