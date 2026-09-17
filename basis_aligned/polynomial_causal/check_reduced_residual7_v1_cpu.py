"""Check newly reduced two-state formula against its old declared-state reference."""
from pathlib import Path
import sys,json,torch
P=Path(__file__).resolve().parent
@torch.no_grad()
def main():
 torch.set_num_threads(2);sys.path.insert(0,str(P/'extracted_circuits/typed_face_residual7_v1'))
 import typed_face_reduced_residual7_v1 as reduced
 import coupled,head8
 from attention8_context_head_terms_v1 import heads
 from head2_mlp8_cross_edit_v1 import retained_delta
 folder=P/'extracted_circuits/typed_face_residual7_v1';p={k:torch.load(folder/file,weights_only=True) for k,file in [('local','local_program.pt'),('context','context_program.pt'),('reentry','reentry_program.pt')]}
 fs=torch.load(P/'TYPED_FACE_RESIDUAL7_V1_ARTIFACT.pt',weights_only=True)['fixtures'];old=torch.load(P/'TYPED_FACE_MLP8_COUPLED_V1_ARTIFACT.pt',weights_only=True)['fixtures'];source=torch.load(P/'MLP8_CONTEXT_SOURCES_V1_ARTIFACT.pt',weights_only=True)['sources'];full=torch.load(P/'extracted_circuits/typed_face_mlp8_coupled_v1/program.pt',weights_only=True);errors=[]
 for f,baseline,s in zip(fs,old,source):
  x=baseline['inputs'];h=heads(p['context'],x['current8'],f['inputs']['token_ids']);complete=coupled.execute(full,**x);delta=.5*head8.execute(full['head8'],x['current8'],x['donor_city8'],x['recipient_token'],x['donor_token'],x['city'],x['destination'])
  target=retained_delta(complete,delta,s[2],h[2],x['post_attention8'],full['mlp8']);actual=reduced.execute(p,**f['inputs']);errors.append(float((actual-target).norm()/target.norm()))
 result={'pred_a':len(errors)==40 and max(errors)<=1e-4,'fixtures':len(errors),'max_two_state_vs_reference_error':max(errors),'scope':'CPU formula equivalence on40opened directly captured residual7 fixtures. Does not replace native effect replay or isolated reduced-package validation. Existing74token vocabulary; fresh confirmation vocabulary expansion pending.'}
 assert result['pred_a'];(P/'REDUCED_RESIDUAL7_V1_CPU_RESULT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
