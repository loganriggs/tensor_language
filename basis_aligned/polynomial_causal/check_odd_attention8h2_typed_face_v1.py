"""CPU identity and saved native behavior audit; no new behavioral promotion."""
import hashlib
import importlib.util
import json
from pathlib import Path
import torch

P = Path(__file__).resolve().parent
D = P / "extracted_circuits/odd_attention8h2_typed_face_v1"
spec = importlib.util.spec_from_file_location("face", D / "execute.py")
face = importlib.util.module_from_spec(spec)
spec.loader.exec_module(face)

def main():
    torch.set_num_threads(2)
    torch.manual_seed(17092051)
    b,t,s,h,d = 2,7,7,5,11
    r0,r1 = [torch.randn(b,t,s,dtype=torch.float64) for _ in range(2)]
    c,i0,i1 = [torch.randn(b,s,h,dtype=torch.float64) for _ in range(3)]
    w = torch.randn(d,h,dtype=torch.float64)
    sm = torch.rand(b,s) > .4
    dm = torch.rand(b,t) > .4
    lam = .37
    args = (r0,r1,c,i0,i1,lam,w,sm,dm)
    actual = face.execute(*args)
    v0 = (1-lam)*c+lam*i0
    v1 = (1-lam)*c+lam*i1
    expected = (((r1*sm[:,None])@v1-(r0*sm[:,None])@v0)@w.T)*dm[...,None]
    state_error = float((actual-expected).norm()/expected.norm())
    self_error = float(face.execute(r0,r0,c,i0,i0,lam,w,sm,dm).abs().max())
    # State atom sums do not commute through a nonlinear suffix; the complete
    # behavioral face nevertheless telescopes to corner5 minus corner0.
    replay = {}
    for stem in ["ODD_ATTENTION8H2_SPARSE_GRAPH_V1", "ODD_ATTENTION8H2_TYPED_FACE_REPLICATION_V1"]:
        path = P/(stem+"_ARTIFACT.pt")
        values = torch.load(path,map_location="cpu",weights_only=True)["values"].double()
        a1=values[1]-values[0];a4=values[4]-values[0]
        a5=values[5]-values[1]-values[4]+values[0]
        direct=face.behavioral_face(values)
        replay[stem] = {"max_abs":float((a1+a4+a5-direct).abs().max()),
                        "artifact_sha256":hashlib.sha256(path.read_bytes()).hexdigest()}
    result = {"state_relative_error":state_error,"self_error":self_error,"native_saved_face_replay":replay,
              "pred_a":state_error<1e-12 and self_error==0 and all(x["max_abs"]<1e-12 for x in replay.values()),
              "scope":"Exact factor-boundary CPU identity and saved native corner telescoping. Fresh package/native replay and selective-removal null remain open.",
              "runtime_sha256":hashlib.sha256((D/"execute.py").read_bytes()).hexdigest()}
    (P/"ODD_ATTENTION8H2_TYPED_FACE_V1_CPU_CONTROL.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))
    assert result["pred_a"]

if __name__ == "__main__":
    main()
