#!/usr/bin/env python3
# BQGATE:0bodyforwards,0textsequences;144cachedendpoints,128permutations;300sec.
"""pred_a replay and deranged multisets; pred_b actual contrast>95% null;
pred_c permuted replacement CE>=.001 all4corpus/mode cells. No fitting.
Null: writer and coefficient marginals explain selective-removal means.
"""
import os,sys,json,time,signal
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal';sys.path.insert(0,str(P))
import torch
import torch.nn.functional as F
from sparse_path_stability_atlas_v1 import digest
STEM='MATCHED_PARTNER_CONTEXT_PERMUTATION_V1'

@torch.no_grad()
def main():
    binding=json.loads((P/(STEM+'_BINDING.json')).read_text())['files'];assert all(digest(k)==v for k,v in binding.items())
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print(json.dumps(dict(body_forwards=0,text_sequences=0,cached_endpoints=144,permutations=128)));return
    signal.alarm(300);tic=time.perf_counter();torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    out=P/(STEM+'_RESULT.json');assert not out.exists()
    data=torch.load(P/'MATCHED_PARTNER_NATURAL_TEXT_V1_PORTS.pt',weights_only=True);rows=torch.load(P/'MATCHED_PARTNER_NATURAL_TEXT_V1_ROWS.pt',weights_only=True)
    program=torch.load(P/'MATCHED_PARTNER_SUBSPACE_V1_PROGRAM.pt',weights_only=True);writer=program['output_writers'][:,8].cuda()
    sd=torch.load(next(k for k in binding if k.endswith('/pytorch_model.bin')),weights_only=True,mmap=True);u=sd['lm_head.weight'].float().cuda()
    h=(data['ports']['pre']+data['ports']['native_output']).cuda();alpha=data['alpha'][:,1].cuda();targets=rows['rows'][:,-1].cuda();meta=data['metadata'];cells=[];checks=[]
    def ce(states,labels):
        parts=[]
        for start in range(0,len(states),8):
            logits=30*torch.tanh(F.linear(F.rms_norm(states[start:start+8],(1152,)),u)/30)
            parts.append(F.cross_entropy(logits.double(),labels[start:start+8],reduction='none'))
        return torch.cat(parts)
    for ci,corpus in enumerate(('fineweb','pile_ood')):
        ids=torch.tensor([i for i,m in enumerate(meta) if m['corpus']==corpus],device='cuda');hh=h[ids];aa=alpha[ids];tt=targets[ids]
        groups=torch.tensor([{'ing':0,'base':1,'other':2}[meta[i]['group']] for i in ids.tolist()],device='cuda');assert len(ids)==72 and torch.equal(groups.cpu(),torch.arange(3).repeat(24))
        baseline=ce(hh,tt);actual=ce(hh-(aa[:,None]*writer).float(),tt)-baseline
        checks.append(float((actual.cpu()-data['scores'][ids.cpu(),2]).abs().max()))
        actual_contrast=float(actual[groups==0].mean()-actual[groups==1].mean());generator=torch.Generator().manual_seed(73110+ci)
        for mode in ('global','within_group'):
            records=[]
            for trial in range(32):
                n=72 if mode=='global' else 24
                while True:
                    perm=torch.randperm(n,generator=generator)
                    if bool((perm!=torch.arange(n)).all()):break
                if mode=='within_group':perm=(3*perm[:,None]+torch.arange(3)[None,:]).flatten()
                assert bool((perm!=torch.arange(72)).all());shuffled=aa[perm.cuda()];assert torch.equal(shuffled.sort().values,aa.sort().values)
                removal=ce(hh-(shuffled[:,None]*writer).float(),tt)-baseline
                replacement=ce(hh+((shuffled-aa)[:,None]*writer).float(),tt)-baseline
                assert bool(torch.isfinite(removal).all()) and bool(torch.isfinite(replacement).all())
                records.append(dict(trial=trial,ing_damage=float(removal[groups==0].mean()),base_damage=float(removal[groups==1].mean()),
                                    contrast=float(removal[groups==0].mean()-removal[groups==1].mean()),replacement_mean=float(replacement.mean()),
                                    replacement_by_group=[float(replacement[groups==g].mean()) for g in range(3)]))
            contrasts=torch.tensor([r['contrast'] for r in records],dtype=torch.float64);replacements=torch.tensor([r['replacement_mean'] for r in records],dtype=torch.float64)
            q95=float(torch.quantile(contrasts,.95));cells.append(dict(corpus=corpus,mode=mode,actual_contrast=actual_contrast,null_contrast_mean=float(contrasts.mean()),null_contrast_q95=q95,
                                actual_exceeds_null=actual_contrast>q95,permuted_replacement_mean=float(replacements.mean()),replacement_damage_pass=float(replacements.mean())>=.001,
                                null_actual_percentile=float((contrasts<actual_contrast).double().mean()),records=records))
    result={'pred_a':max(checks)<=1e-5,'pred_b':all(c['actual_exceeds_null'] for c in cells),'pred_c':all(c['replacement_damage_pass'] for c in cells)}
    result.update(cells=cells,replay_errors=checks,source_shas=binding,execution_seconds=time.perf_counter()-tic,
                  scope='Frozen writer/coefficient permutation diagnostic; no data fitting. Failure narrows context-coupling interpretation of natural removal selectivity, not existence of weight structure.')
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(**{k:v for k,v in result.items() if k not in ('cells','source_shas')},cells=[{k:v for k,v in c.items() if k!='records'} for c in cells])),flush=True)

if __name__=='__main__':main()
