"""Exact fused-operator equivalence and bounded CPU timing, not GPU speed claim."""
from pathlib import Path
import sys,json,time,torch,statistics
P=Path(__file__).resolve().parent
@torch.no_grad()
def main():
 torch.set_num_threads(2);folder=P/'extracted_circuits/typed_face_reduced_residual7_v1';sys.path.insert(0,str(folder));import execute as original
 import typed_face_reduced_fused_v1 as fused
 program={k:torch.load(folder/file,weights_only=True) for k,file in [('local','local_program.pt'),('context','context_program.pt'),('reentry','reentry_program.pt')]};fixtures=torch.load(P/'TYPED_FACE_REDUCED_RESIDUAL7_V1_ARTIFACT.pt',weights_only=True)['fixtures'];errors=[]
 for f in fixtures:
  a=original.execute(program,**f['inputs']);b=fused.execute(program,**f['inputs']);errors.append(float((a-b).norm()/a.norm()))
 x=fixtures[0]['inputs'];original.execute(program,**x);fused.execute(program,**x);times={'original':[],'fused':[]}
 for repeat in range(6):
  order=[('original',original),('fused',fused)] if repeat%2==0 else [('fused',fused),('original',original)]
  for name,module in order:
   start=time.perf_counter();module.execute(program,**fixtures[repeat]['inputs']);times[name].append(time.perf_counter()-start)
 result={'pred_a':len(errors)==40 and max(errors)<=1e-10,'max_formula_error':max(errors),'fixture_count':40,'operation_counts':{'full_attention8':{'before':2,'after':1},'head8_delta':{'before':2,'after':1},'mlp8_down':{'before':4,'after':1}},'cpu_seconds':times,'cpu_median_seconds':{k:statistics.median(v) for k,v in times.items()},'static_float_scalars':24153091,'scope':'FP64-algebra equivalence on40opened native-state fixtures; unchanged weights and counterfactual. CPU timing uses2threads,sixinterleavedexamples afterwarmup; not a GPU or end-to-end speedup claim. Native and isolated fused-package validation pending.'}
 assert result['pred_a'];(P/'REDUCED_FUSED_V1_CPU_RESULT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
