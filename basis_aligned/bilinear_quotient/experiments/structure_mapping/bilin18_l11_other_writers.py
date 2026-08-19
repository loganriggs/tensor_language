"""User prediction (2026-08-17): L11 -- the dissident that shares none of the
16-reader vocabulary over writer L1 (R^2 -0.10, section 84) -- "reads from other
layers": its functionals should reconstruct from the shared code of some OTHER
writer. Every dissident measurement so far used L1's output coordinates only.

For writers W in (0, 9): build the six-reader coupling family over W's top-48
output-PCA coords (readers 2,3,5,13,17 minus any equal to W, plus 15), fit the
top-80 basis, reconstruct L11's 40 functionals over W. REGISTERED (user's):
(a) L11 median R^2 >= 0.4 for at least one W; control (b): a held-out
non-dissident reader (L12) reconstructs at >= 0.5 for the same W (the basis
itself is healthy); null (c): random symmetric matrices <= 0.15."""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
from bilin18_joint_removal import fwd, orth, m, FW, DEV
D,K,NF=1152,48,40
OUT=('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
     'bilin18_l11_other_writers_results.json')

@torch.no_grad()
def main():
    t0=time.time()
    def collect(li):
        outs=[]
        h=m.transformer.h[li].mlp.register_forward_hook(
            lambda mo_,i_,o_: outs.append(o_.detach().reshape(-1,D).float()))
        for i in range(0,60,6):
            b=FW[i:i+6,:513].to(DEV)
            m(b[:,:-1].contiguous(), b[:,1:].contiguous())
        h.remove()
        return torch.cat(outs)
    def family(writer, readers):
        Yw=collect(writer); Yc=Yw-Yw.mean(0)
        _,_,Vh=torch.linalg.svd(Yc, full_matrices=False)
        V=orth(Vh[:K].T)
        rows={}
        for j in readers:
            Yj=collect(j)
            _,_,Vhj=torch.linalg.svd((Yj-Yj.mean(0)).float(), full_matrices=False)
            P=orth(Vhj[:NF].T)
            mlp=m.transformer.h[j].mlp
            L=mlp.Left.weight.detach().float()@V
            R=mlp.Right.weight.detach().float()@V
            DwP=mlp.Down.weight.detach().float().T@P
            rs=[]
            for f in range(NF):
                M=torch.einsum('k,ka,kb->ab',DwP[:,f],L,R)
                Ms=0.5*(M+M.T)
                rs.append((Ms/Ms.norm().clamp_min(1e-12)).flatten())
            rows[j]=torch.stack(rs)
        return rows
    g=torch.Generator(device=DEV).manual_seed(0)
    out={}
    for W in (0,9):
        readers=[r for r in (2,3,5,13,15,17) if r!=W]
        fams=family(W, readers+[11,12])
        X=torch.cat([fams[r] for r in readers])
        _,_,Wsvd=torch.linalg.svd(X, full_matrices=False)
        B=Wsvd[:80]
        def med_r2(rows):
            r2s=[]
            for v in rows:
                rec=(v@B.T)@B
                r2s.append(1-float(((v-rec)**2).sum()))
            return sorted(r2s)[len(r2s)//2]
        r11=med_r2(fams[11]); r12=med_r2(fams[12])
        rnd=[]
        for _ in range(NF):
            A=torch.randn(K,K,device=DEV,generator=g); A=0.5*(A+A.T)
            rnd.append((A/A.norm()).flatten())
        rr=med_r2(torch.stack(rnd))
        out[f'writer{W}']={'L11':r11,'L12_ctrl':r12,'random':rr}
        print(f'writer L{W}: L11 median R^2 {r11:+.2f} | L12 control {r12:+.2f} | '
              f'random {rr:+.2f}',flush=True)
    pa=any(out[w]['L11']>=0.4 for w in out)
    pb=any(out[w]['L12_ctrl']>=0.5 for w in out)
    pc=all(out[w]['random']<=0.15 for w in out)
    out['pred_a_user']=bool(pa); out['ctrl_b']=bool(pb); out['null_c']=bool(pc)
    print(f"\n(a) [user] L11 shares some writer's code (>=0.4): {'HELD' if pa else 'FAILED'}")
    print(f"(b) control reader healthy (>=0.5): {'HELD' if pb else 'VIOLATED'}")
    print(f"(c) random null (<=0.15): {'HELD' if pc else 'VIOLATED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
