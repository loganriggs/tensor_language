"""Export the coupled direct-plus-MLP8 value operator."""
from pathlib import Path
import hashlib,json,shutil
P=Path(__file__).resolve().parent
def main():
 target=P/'extracted_circuits/city_mlp8_coupled_value_v1';target.mkdir(parents=True,exist_ok=True)
 shutil.copy2(P/'mlp8_coupled_value_v1.py',target/'execute.py');shutil.copy2(P/'mlp8_value_norm_closed_v1.py',target/'mlp8_value_norm_closed_v1.py');shutil.copy2(P/'CITY_MLP8_COUPLED_VALUE_V1_PROGRAM.pt',target/'program.pt')
 local=json.loads((P/'CITY_MLP8_COUPLED_VALUE_V1_CPU_RESULT.json').read_text())
 manifest={'schema':'regional.city_mlp8_coupled_value.v1','name':target.name,'certified_complete':False,'ports':['z8[1,T,1152] native unnormalized MLP8 input','delta[1,T,1152] upstream city intervention','token_ids[1,T]'],'native_state_scalars_T32':36864,'intervention_scalars_T32':36864,'static_float_scalars':local['stored_float_scalars'],'weight_bytes':local['stored_float_bytes'],'token_id_bytes':local['stored_integer_bytes'],'supported_sequence_tokens':local['tokens'],'derived_cache_bytes':local['derived_folded_down_bytes'],'external_computation':'Native z8/upstream delta generation and later suffix; not token-only.','four_properties':{'ood_prediction':'fresh V1 and V2 coupled screens pass globally','extraction':True,'selective_manipulation':'fresh same-boundary norm-matched null passes for MLP8 mediator; coupled operator inherits fresh full-suffix controls','composition_reuse':'candidate coupled operator; independent subterm composition unresolved'},'simplicity':'No matched-effect simplicity claim; 16,882,563 stored FP32 values plus native ports.','evidence':{'local':'../../CITY_MLP8_COUPLED_VALUE_V1_CPU_RESULT.json','fresh_v1':'../../CITY_VALUE_PATH_FRESH_V1_RESULT.json','fresh_v2':'../../CITY_VALUE_PATH_FRESH_V2_RESULT.json','fresh_null':'../../CITY_MLP8_NORM_CLOSED_FRESH_V1_NULL_RESULT.json'},'files':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in target.iterdir() if f.suffix in ['.py','.pt']}}
 (target/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
if __name__=='__main__':main()
