"""Package the exact reduced-input, fresh-confirmed MLP8 value mediator."""
import hashlib,json,shutil
from pathlib import Path
P=Path(__file__).resolve().parent

def main():
 target=P/'extracted_circuits/city_mlp8_value_mediator_v1';target.mkdir(exist_ok=True)
 shutil.copyfile(P/'mlp8_value_mediator_v1.py',target/'execute.py');shutil.copyfile(P/'CITY_MLP8_VALUE_EXTRACTED_V1_PROGRAM.pt',target/'program.pt')
 fresh=json.loads((P/'CITY_VALUE_MEDIATION_FRESH_V1_RESULT.json').read_text());installed=json.loads((P/'CITY_MLP8_VALUE_EXTRACTED_INSTALLED_V1_RESULT.json').read_text())
 isolated=P/'CITY_MLP8_VALUE_EXTRACTED_V1_ISOLATED_RESULT.json';iso=json.loads(isolated.read_text())['passes'] if isolated.exists() else None
 m={'schema':'regional.city_mlp8_value_mediator.v1','name':target.name,'certified_complete':False,'counterfactual':'Return MLP8-induced head9.8 current-value correction, to be subtracted while retaining the exact native upstream head8.2 city swap. This is not whole-MLP8 ablation.','ports':['z8[1,T,1152] native unnormalized MLP8 input','delta[1,T,1152] upstream attention8 intervention','edited_rho9[1,T,1] native edited mixed9 RMS'],'native_vector_state_arrays':1,'native_scalar_state_arrays':1,'intervention_vector_arrays':1,'native_state_scalars_T32':36896,'intervention_scalars_T32':36864,'static_float_scalars':11206658,'weight_bytes':44826632,'external_computation':'Native z8 and edited RMS9 construction, upstream native cityswap and fullsuffix. No token-only or independent composition claim.','four_traits':{'ood_prediction':fresh['pred_a'] and fresh['pred_b'],'extraction':iso and all(installed['pred_'+k] for k in 'abc') if iso is not None else None,'selective_removal':None,'selective_manipulation':all(fresh['pred_'+k] for k in 'acde'),'composition_reuse':None},'property_scope':'Fresh same-corpus FineWeb mediator test after opened FineWeb selection; fixed probes. Conditional native-context formula predicts native correction. Prior original removal/swap direction failures and opened subgroup leakage retained; no structural/domain-universal guarantee.','evidence':{'fresh':'../../CITY_VALUE_MEDIATION_FRESH_V1_RESULT.json','opened_specificity':'../../CITY_FINEWEB_VALUE_NULL_V1_RESULT.json','reduced_inputs':'../../CITY_MLP8_VALUE_EXTRACTED_V1_CPU_RESULT.json','installed':'../../CITY_MLP8_VALUE_EXTRACTED_INSTALLED_V1_RESULT.json','isolated':'../../CITY_MLP8_VALUE_EXTRACTED_V1_ISOLATED_RESULT.json'},'files':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in target.iterdir() if f.suffix in ['.py','.pt']}}
 (target/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
 (target/'README.md').write_text('''# MLP8-mediated current-value correction

Load program.pt with torch.load(weights_only=True). Import execute from this folder and call `execute.execute(program,z,delta,edited_rho9)`.

CPU batch1 interface: z and delta[1,T,1152] are native unnormalized MLP8 input and the attention8 edit; edited_rho9[1,T,1] is the edited block9 mixed-residual RMS (FP64, including native FP32epsilon). Return FP64[1,T,128]. Subtract this correction from head9.8 value while retaining the upstream cityswap. Both keys and other heads remain untouched; fullsuffix must recompute. This is not native MLP8 ablation.

Only PyTorch and these two files are required. Invalid norm shape, nonpositive/nonfinite norms, non-CPU input and batch>1 fail. All ordered MLP8 cross terms, quadratic and its normalization change remain coupled. Edited RMS9 remains an external native scalar; the program does not infer it from tokens.

Storage:11,206,658FP32 values,44,826,632bytes. AtT32:36,896 native scalars (one z8 array plus32editedRMS values), plus36,864 intervention scalars. This eliminates unused mixed9 arrays and147,456 weight values from the earlier six-piece helper. It does not eliminate the edited-RMS dependency or price the full external model as free.

Fresh mediator prediction, globalpreservation/direction/null gates pass; independent composition and matched-effect simplicity remain unresolved. See manifest for actual isolated/installed status and primary receipts. Earlier FineWeb failures for the original cityremoval/swap remain in the research record.
''')
if __name__=='__main__':main()
