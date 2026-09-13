"""Exact receipt-aligned coverage of original two-edit behavior by local program."""
import json,hashlib
from pathlib import Path
import torch
P=Path(__file__).resolve().parent

def main():
    source=P/'MLP9_TO_MLP10_RESIDUAL_FOLD_V1_ARTIFACT.pt';new=P/'COMPOSED_BACKGROUND_SHARED_V2_RESULT.json'
    a=torch.load(source,weights_only=True)['measures'].double();r=json.loads(new.read_text());arr=lambda key:torch.tensor([c[key] for c in r['cells']],dtype=torch.float64)
    nativecross=arr('native_effects');assert torch.equal(nativecross,a[:,8]-a[:,5])
    original=a[:,2]-a[:,1]-a[:,3]+a[:,0]
    before10=a[:,2]-a[:,4];block10=a[:,4]-a[:,5];suffix=a[:,5]-a[:,1]-a[:,3]+a[:,0]
    identity=float((before10+block10+suffix-original).abs().max());assert identity<1e-12
    genbaseline=a[:,5]+arr('background_drift');genparent=genbaseline+arr('transported_effects')
    predicted=genparent-a[:,1]-a[:,3]+a[:,0]
    missing_local=block10-nativecross
    rows=[]
    for lo,hi,label in [(24*k,24*(k+1),'regional'+str(k)) for k in range(4)]+[(96+16*k,112+16*k,'fineweb'+str(k)) for k in range(4)]:
        cells={}
        target=original[lo:hi]
        for name,x in [('before10',before10),('block10_total',block10),('suffix_baseline',suffix),('local_cross',nativecross),('unmodeled_block10',missing_local),('current_predicted_interaction',predicted)]:
            x=x[lo:hi];den=target.square().sum(0).clamp_min(1e-30)
            cells[name]=dict(norm_over_original=(x.norm(dim=0)/den.sqrt()).tolist(),signed_projection=((x*target).sum(0)/den).tolist())
        diff=predicted[lo:hi]-target
        rows.append(dict(group=label,original_effect_norm=target.norm(dim=0).tolist(),prediction_relative_error=(diff.norm(dim=0)/target.norm(dim=0).clamp_min(1e-30)).tolist(),opposite_signs=(predicted[lo:hi]*target<0).sum(0).tolist(),decomposition=cells))
    result=dict(reference_cross_bitwise_replay=True,telescoping_maxabs=identity,groups=rows,
        sources={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in [source,new]},
        scope='Historical native endpoint receipt identity and signed projections. Current parent candidate = generated additive postMLP10 background + generated fullcross, then native suffix. Telescoping differences are path-specific, not invariant causal shares. No fitting/newforwards.')
    (P/'COMPOSED_ORIGINAL_INTERACTION_COVERAGE_V1_RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
    for c in rows:print(c['group'],'error',c['prediction_relative_error'],'targetprojections',{k:v['signed_projection'][0] for k,v in c['decomposition'].items()})
if __name__=='__main__':main()
