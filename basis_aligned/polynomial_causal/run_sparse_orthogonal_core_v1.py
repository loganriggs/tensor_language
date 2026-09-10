"""Weight-only centered tensor: spectral128inputbasis, exact256edge selection.

Registered onboard: Atrace/core identities<=1e-10; Bcapture>=.05; C>128offdiagonal
edges andmaxnodeincidentenergyfraction<=.25. Basis remainsunoptimized.
"""
from pathlib import Path
import json,time,hashlib,torch
from sparse_orthogonal_quadratic_core_v1 import input_marginal,orthogonal_core
P=Path(__file__).resolve().parent
CK=Path('/workspace/.hf_home/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240/pytorch_model.bin')

def main():
    assert not torch.cuda.is_available(),'RunCPUonlywithCUDA_VISIBLE_DEVICES empty'
    assert json.loads((P/'SPARSE_ORTHOGONAL_CORE_V1_CONTROL.json').read_text())['passed']
    out=P/'SPARSE_ORTHOGONAL_CORE_V1_RESULT.json';assert not out.exists();tic=time.perf_counter();torch.set_num_threads(2)
    sd=torch.load(CK,map_location='cpu',weights_only=True,mmap=True);u=sd['lm_head.weight'].double();mean=u.mean(0);metric=u.T@u-len(u)*torch.outer(mean,mean);del u
    root=torch.linalg.cholesky(metric).T
    l,r,d=[sd[f'transformer.h.17.mlp.{k}.weight'].double() for k in ['Left','Right','Down']];z=root@d
    print('Computing centered native input marginal',flush=True)
    marginal=input_marginal(l,r,z.T@z);total=marginal.trace();ev,vec=torch.linalg.eigh(marginal);basis=vec[:,-128:].T
    fulltotal=json.loads((P/'FULLU_OUTPUT_FUNCTIONS_V1_AUDIT.json').read_text())['native_total'];centerfraction=json.loads((P/'COMMON_OUTPUT_QUADRATIC_V1_AUDIT.json').read_text())['native_centered_energy_fraction'];bridge=abs(float(total)/(fulltotal*centerfraction)-1)
    assert bridge<=1e-10 and float(ev.min())>=-1e-10*float(total)
    print('Projecting orthogonal quadratic features',flush=True)
    coefficients,edges=orthogonal_core(l,r,z,basis);energies=coefficients.square().sum(0);selected=energies.topk(256).indices
    kept=edges[:,selected];captured=float(energies[selected].sum()/total)
    physical=torch.linalg.solve_triangular(root,coefficients[:,selected],upper=True)
    writerbridge=float((root@physical-coefficients[:,selected]).norm()/coefficients[:,selected].norm())
    ortherror=float((basis@basis.T-torch.eye(128,dtype=basis.dtype)).abs().max());assert max(writerbridge,ortherror)<=1e-10
    incident=torch.zeros(128,dtype=basis.dtype)
    for k,energy in zip(kept.T,energies[selected]):
        incident[k[0]]+=energy
        if k[1]!=k[0]:incident[k[1]]+=energy
    maxincident=float(incident.max()/energies[selected].sum());off=int((kept[0]!=kept[1]).sum())
    ap=P/'SPARSE_ORTHOGONAL_CORE_V1_CENTERED.pt';assert not ap.exists();torch.save(dict(basis=basis,edges=kept,writer=physical,unembedding_mean=mean,scope='Centeredcomponentonly. Preserve exactcommonquadratic separately; no fullmodeladoption.'),ap)
    result=dict(predictions={'pred_a_instrument':max(bridge,writerbridge,ortherror)<=1e-10,'pred_b_centered_capture':captured>=.05,'pred_c_distributed_interaction_graph':off>128 and maxincident<=.25},centered_coefficient_capture=captured,full_projected_core_capture=float(energies.sum()/total),input_one_mode_top128_energy=float(ev[-128:].sum()/total),active_input_readers=int(torch.unique(kept).numel()),offdiagonal_edges=off,maximum_node_incident_energy_fraction=maxincident,top_node_incident_energy_fractions=[float(v) for v in incident.topk(8).values/energies[selected].sum()],native_centered_total=float(total),native_trace_bridge=bridge,writer_bridge=writerbridge,basis_orthogonality_error=ortherror,price=dict(body_forwards=0,basis_float_numbers=128*1152,writer_float_numbers=256*1152,edge_indices=512,centered_float_numbers=442368,additional_exact_common_symmetric_numbers=1152*1153//2,checkpoint_bytes=ap.stat().st_size),checkpoint_sha256=hashlib.sha256(ap.read_bytes()).hexdigest(),wall_seconds=time.perf_counter()-tic,scope='Conditional exact sparseedge choice in a fixedorthogonal spectralinputbasis; basisnotoptimized. Outputcenteredmetric, explicitcommonchannelpreservedseparately. Notglobalbestbasis, circuit, or fullmodeladoption.')
    with out.open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result),flush=True)
if __name__=='__main__':main()
