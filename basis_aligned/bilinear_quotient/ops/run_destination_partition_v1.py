#!/usr/bin/env python3
# BQGATE:880bodyforwards;40prefixes;300seconds;no fitting.
"""pred_a anchors/write sum<=1e-5; pred_b both pieces>=.05full and1e-5RMS.
pred_c interaction<=.25smallersingle and<=.10full; pred_d<=.5nullmedian,beats8/9.
880forwards. Opened source-partition composition/null screen, not selective units.
"""
import hashlib,json,os,signal,sys,time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal'
sys.path[:0]=[str(Path(__file__).parent),str(P),str(ROOT)]
from regional_endpoint_batching_v1 import group_rows,expand
from regional_paired_write_runtime_v3 import measure
from run_even_value_factorial_native_v1 import setup
from typed_face_write_atoms_v1 import native
STEM='DESTINATION_PARTITION_V1';ARMS=['native','full']+[f'{part}:{side}' for part in range(10) for side in ['A','B']]
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
@torch.no_grad()
def main():
 binding=json.loads((P/(STEM+'_BINDING.json')).read_text())['files'];assert all(digest(k)==v for k,v in binding.items())
 control=json.loads((P/(STEM+'_CPU_CONTROL.json')).read_text());assert control['pred_a'] and control['uniform_null_count']==9
 doc=json.loads((P/'TYPED_FACE_PROSPECTIVE_V1_ROWS.json').read_text());rows=doc['rows'];groups,mapping=group_rows(rows)
 masks={(r['context_id'],r['cue']):r for r in control['records']}
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
  assert len(groups)==40 and len(ARMS)==22;print('880bodyforwards;40prefixes;semantic and nine matched partitions');return
 out=P/(STEM+'_RESULT.json');assert not out.exists();start=time.perf_counter();signal.alarm(300);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
 from fastload import load_model_fast
 model=load_model_fast().cuda().eval();graph,_,_,_=setup('cuda');p={k:v.to('cuda') for k,v in torch.load(P/'TYPED_FACE_KEY_SOURCE_FRESH_V1_PROGRAM.pt',weights_only=True).items()}
 state={};writes={a:[] for a in ARMS[1:]}
 def write(arm,row,donor_row,current,donor,mask):
  chosen=mask.clone()
  if arm!='full':
   part,side=arm.split(':');part=int(part);record=masks[(row['context_id'],row['cue'])];a=record['framing'] if part==0 else record['nulls'][part-1]
   indices=a if side=='A' else [i for i in record['full'] if i not in a];chosen.zero_();chosen[indices]=True
  city=row['city_position'];return .5*native.execute(p,current,donor[:,city],row['ids'][city],donor_row['ids'][city],city,chosen)
 def observe(arm,g,raw,delta,lam,mask,original):writes[arm].append(original.detach().cpu());return original
 m=measure(model,graph,groups,ARMS,write,observe);v=expand(m['values'],mapping,6);e=v-v[:1]
 old=torch.load(P/'TYPED_FACE_PROSPECTIVE_V1_ARTIFACT.pt',weights_only=True)['values'][[0,3]];anchor=float((v[:2]-old).abs().max())
 write_errors=[]
 for part in range(10):
  for a,b,f in zip(writes[f'{part}:A'],writes[f'{part}:B'],writes['full']):write_errors.append(float((a+b-f).norm()/f.norm()))
 records=[]
 for part in range(10):
  a=e[2+2*part];b=e[3+2*part];joint=e[1];interaction=joint-a-b;families={}
  for family in doc['variants']:
   idx=[i for i,r in enumerate(rows) if r['variant']==family];an=a[idx,0].norm();bn=b[idx,0].norm();jn=joint[idx,0].norm();inter=interaction[idx,0].norm()
   families[family]={'interaction_over_smaller_single':float(inter/torch.minimum(an,bn).clamp_min(1e-8)),'joint_from_singles_error':float(inter/jn.clamp_min(1e-8)),
    'piece_rms_logits':[float(a[idx,0].square().mean().sqrt()),float(b[idx,0].square().mean().sqrt())],'piece_over_full':[float(an/jn),float(bn/jn)],
    'all_readout_piece_rms':torch.stack([a[idx].square().mean(0).sqrt(),b[idx].square().mean(0).sqrt()]).tolist(),'all_readout_interaction_rms':interaction[idx].square().mean(0).sqrt().tolist()}
  records.append({'partition':part,'families':families,'worst_interaction':max(x['interaction_over_smaller_single'] for x in families.values())})
 actual=records[0];null=torch.tensor([r['worst_interaction'] for r in records[1:]],dtype=torch.float64);median=float(null.median());ratio=actual['worst_interaction']/max(median,1e-30)
 result={'pred_a':anchor<=1e-5 and max(write_errors)<=1e-5 and bool(torch.isfinite(v).all()) and m['body_forwards']==880,
 'pred_b':all(min(x['piece_rms_logits'])>=1e-5 and min(x['piece_over_full'])>=.05 for x in actual['families'].values()),
 'pred_c':all(x['interaction_over_smaller_single']<=.25 and x['joint_from_singles_error']<=.1 for x in actual['families'].values()),
 'pred_d':ratio<=.5 and int((null>actual['worst_interaction']).sum())>=8,
 'canonical':actual,'random_partitions':records[1:],'canonical_over_null_median':ratio,'nulls_beaten':int((null>actual['worst_interaction']).sum()),'anchor_max_abs':anchor,'max_write_partition_error':max(write_errors),'body_forwards':m['body_forwards'],'seconds':time.perf_counter()-start,'source_shas':binding,
 'scope':'Opened framing/clause source-destination partition of complete retained write. Matched-cardinality random partitions, not physical-product split. Per-piece selective removal and fresh confirmation untested; prior failures unchanged.'}
 torch.save({'values':v},P/(STEM+'_ARTIFACT.pt'));out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ['source_shas','random_partitions','canonical']},indent=2));signal.alarm(0)
if __name__=='__main__':main()
