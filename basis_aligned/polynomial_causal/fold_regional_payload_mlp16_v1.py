"""Exact conditional MLP16 producer for the two regional source-child readings."""
from pathlib import Path
import json,torch
from sparse_path_stability_atlas_v1 import digest
P=Path(__file__).resolve().parent
@torch.no_grad()
def main():
    torch.set_num_threads(2);binding=json.loads((P/'REGIONAL_CROSS_START_NATIVE_V1_BINDING.json').read_text())['files'];ck=next(k for k in binding if k.endswith('pytorch_model.bin'));state=torch.load(ck,weights_only=True,mmap=True,map_location='cpu')
    source=P/'COMMON_QUADRATIC_SOURCE_BLOCK_V1_ARTIFACT.pt';atoms=torch.load(source,weights_only=True,map_location='cpu')['atoms'];readers=atoms[-2:,2];current=readers[:,:1152];first=readers[:,1152:]
    left,right,down=[state['transformer.h.16.mlp.'+n+'.weight'].double() for n in ('Left','Right','Down')];bias=state['transformer.h.16.mlp.Down_bias'].double();lambdas=state['transformer.h.17.lambdas'].double();folded=lambdas[0]*current@down;folded_bias=lambdas[0]*(current@bias)
    prior=torch.load(P/'SHARED_NODE_CANONICAL_BRANCHES_V1_SPECTRAL.pt',weights_only=True,map_location='cpu')['nodes'][1];old=torch.cat([prior['reader'][None],prior['partners'].T],0).double();new_basis=torch.linalg.qr(current.T,mode='reduced')[0];old_basis=torch.linalg.qr(old.T,mode='reduced')[0];overlap=torch.linalg.svdvals(new_basis.T@old_basis)
    torch.manual_seed(73500);eps=torch.finfo(torch.float32).eps;x=torch.randn(32,1152,dtype=torch.float64);x=x/(x.square().mean(-1,keepdim=True)+eps).sqrt();back=torch.randn_like(x);base=torch.randn_like(x);base=base/(base.square().mean(-1,keepdim=True)+eps).sqrt();products=(x@left.T)*(x@right.T);native_mlp=products@down.T+bias;r17=back+lambdas[0]*native_mlp;rho=(r17.square().mean(-1,keepdim=True)+eps).sqrt()
    direct=(r17/rho)@current.T+base@first.T;compiled=(back@current.T+products@folded.T+folded_bias)/rho+base@first.T;error=float((direct-compiled).norm()/direct.norm());assert error<=1e-12
    art=P/'REGIONAL_PAYLOAD_MLP16_FOLD_V1_ARTIFACT.pt';out=P/'REGIONAL_PAYLOAD_MLP16_FOLD_V1_RESULT.json';assert not art.exists() and not out.exists();torch.save(dict(current_readers=current,first_readers=first,folded_down=folded,folded_bias=folded_bias,block17_lambdas=lambdas),art)
    result=dict(relative_replay_error=error,prior_parent1_reader_span_cosines=overlap.tolist(),folded_output_weights=list(folded.shape),artifact_sha=digest(art),source_artifact_sha=digest(source),checkpoint_sha=binding[ck],scope='Exact conditional producer of regional2Dpayload. MLP16Left/Right, actualRMS17 divisor, otherincomingresidual andfirst-attention input remain explicit; no native causal contribution measured yet. Existingproducer algebra reused on a new frozen semantic component.')
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
