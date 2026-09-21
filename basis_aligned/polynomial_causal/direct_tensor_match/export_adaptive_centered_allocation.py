"""Reuse existing slice factors; allocate at most two products per output mode."""
from pathlib import Path
import json,torch
from adaptive_output_rank_budget import allocate,toy_check
from midpoint_program import product_source_delta

def main():
    torch.set_num_threads(2);torch.set_grad_enabled(False);p=Path(__file__).resolve().parent
    out=p/'MIDPOINT_ADAPTIVE_ALLOCATION_V1.json';assert not out.exists()
    result=json.loads((p/'MIDPOINT_CENTERED_ALLOCATION_V1.json').read_text())
    assert result['predictions']['pred_a_instrument']
    spectra=result['spectra']
    energy=torch.tensor([[v['total_energy']-v['tail1'],v['tail1']-v['tail2']] for v in spectra],dtype=torch.float64).clamp_min(0)
    ranks,retained=allocate(energy,512)
    selected=torch.tensor([2*j+k for j,r in enumerate(ranks.tolist()) for k in range(r)],dtype=torch.long)
    graphs=torch.load(p/'MIDPOINT_CENTERED_ALLOCATION_GRAPHS_V1.pt',weights_only=True);source=graphs['w512r2']
    e={k:source[k] for k in ['linear_n','linear_m','full_mean']}
    for key in ['A','B']:e[key]=source[key][:,selected]
    for key in ['base_left_mean','base_right_mean','product_mean']:e[key]=source[key][selected]
    e['reduced_writers']=source['group_writers'][:,selected//2]
    rows=torch.load(p/'MIDPOINT_CALIBRATION_ROWS_V1.pt',weights_only=True)
    n=rows['n'].flatten(0,1)[:64].double();m=rows['m'].flatten(0,1)[:64].double();dm=m.roll(7,0)-m
    fullleft=n@source['A']-source['base_left_mean'];fullright=dm@source['B']
    mask=torch.zeros(1024,dtype=torch.float64);mask[selected]=1
    expected=((fullleft*fullright*mask).reshape(64,512,2).sum(-1))@source['group_writers'].T+dm@source['linear_m']
    actual=product_source_delta(e,n,dm)
    replay=float((actual-expected).norm()/expected.norm());assert replay<1e-12
    programs={k:graphs[k] for k in ['prior512','w256r2','w512r1']};programs['adaptive512']=e
    path=p/'MIDPOINT_ADAPTIVE_ALLOCATION_GRAPHS_V1.pt';assert not path.exists();torch.save(programs,path)
    info=dict(toy=toy_check(),replay=replay,ranks=ranks.tolist(),selected_indices=selected.tolist(),retained_energy=retained,wide_uniform_energy=float(energy[:,0].sum()),narrow_uniform_energy=float(energy[:256].sum()),active_output_directions=int((ranks>0).sum()),weight_coefficients=2*1152*512+1152*512+2*1152**2,mean_state=2*512+1152,scope='Optimal retained slice energy for fixed512products and candidate first2singular terms in each of512orthogonal outputs. Output writers explicitly duplicated for two-product groups and fully charged. No claim of optimal storage or native error. No SVD recomputation or held fitting.')
    out.write_text(json.dumps(info,indent=2)+'\n');print(json.dumps({k:v for k,v in info.items() if k not in ['ranks','selected_indices']},indent=2))
if __name__=='__main__':main()
