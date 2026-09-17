"""Price exact native weight sharing in the widened local executor."""
from pathlib import Path
import torch,json
P=Path(__file__).resolve().parent

def main():
 folder=P/'extracted_circuits/typed_face_context_generated_v1';local=torch.load(folder/'local_program.pt',weights_only=True,map_location='cpu');context=torch.load(folder/'context_program.pt',weights_only=True,map_location='cpu');head=local['head8'];sl=slice(256,384);checks={};saved=0
 for h,c in [('q1','q1'),('k1','k1'),('q2','q2'),('k2','k2'),('current_value','value'),('output','output'),('mixture','mixture')]:
  target=context[c] if h=='mixture' else context[c][:,sl] if h=='output' else context[c][sl]
  checks[h]=torch.equal(head[h],target);saved+=head[h].numel()
 total=sum(v.numel() for d in local.values() for v in d.values() if v.is_floating_point())+sum(v.numel() for v in context.values() if v.is_floating_point())
 result={'pred_a':all(checks.values()),'exact_equal_slices':checks,'duplicated_float_scalars':saved,'current_float_scalars':total,'deduplicated_float_scalars':total-saved,'scope':'Exact static tensor-equality audit only. No package mutation or new native execution. Sharing preserves the mathematical operator; physical layout replay and isolated v2 bundle remain necessary. Not evidence for circuit simplicity against matched-effect random components.'}
 assert result['pred_a'];(P/'GENERATED_CONTEXT_SHARING_V1_RESULT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
