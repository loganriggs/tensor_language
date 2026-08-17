"""Does the syntax bus have an origin? The L16->L17 edge is the model's strongest
verified interaction (product law, +0.067 nats under cut-and-finetune), and the
report lists its origin as unknown. Sections 92-93 say generic tail routing is
unaimed and dilution-priced. Question: is the CONTENT L16 forwards along its top-8
bus span accumulated diffusely by that same law, or does some upstream layer feed
the bus specifically?

For each source layer s in 5..15: transplant s's full MLP write (different
documents), measure the induced movement of L16's write ALONG THE BUS SPAN,
normalized by the bus coordinates' variance. Null model: movement proportional to
s's dilution share (write-to-stream ratio, section 93) times its generic effect on
L16 -- i.e. the bus is fed like everything else.

REGISTERED PREDICTIONS: (a) diffuse -- every layer's bus movement is within 2x of
its dilution-share expectation (bus content accumulates like generic content; the
bus is a channel, not a station with a dedicated supplier); (b) alternative -- if
some source exceeds 3x its expectation, the bus has an origin and it should be
L15 or L14 (adjacent, per the one-layer coherence length). Control: same
measurement on a random 8-dim span of L16's output (matched construction) --
expectations should hold there regardless."""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
from bilin18_joint_removal import fwd, orth, m, FW, DEV
D=1152
OUT=('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
     'bilin18_bus_origin_results.json')

@torch.no_grad()
def main():
    t0=time.time()
    base_rows=FW[300:324,:257].to(DEV); src_rows=FW[400:424,:257].to(DEV)
    # bus span: L16's top-8 output-PCA directions (as in the composition-law arc)
    accs=[]
    for i in range(0,60,6):
        acc=[]; fwd(FW[i:i+6,:513].to(DEV), collect=16, acc=acc); accs.append(acc[0])
    Y16=torch.cat(accs)
    _,_,Vh=torch.linalg.svd((Y16-Y16.mean(0)).float(), full_matrices=False)
    BUS=orth(Vh[:8].T)
    g=torch.Generator(device=DEV).manual_seed(0)
    RND=orth(torch.randn(D,8,device=DEV,generator=g))
    # dilution shares of sources wrt the stream entering L16
    mos={}; res16=[]
    hs=[m.transformer.h[16].register_forward_pre_hook(
        lambda mod,inp: res16.append(inp[0].detach().reshape(-1,D).float()) or None)]
    for li in range(5,16):
        def mko(li=li):
            return lambda mod,i_,o_: mos.setdefault(li,[]).append(
                o_.detach().reshape(-1,D).float())
        hs.append(m.transformer.h[li].mlp.register_forward_hook(mko()))
    for i in range(0,36,6):
        b=FW[i:i+6,:513].to(DEV)
        m(b[:,:-1].contiguous(), b[:,1:].contiguous())
    for h in hs: h.remove()
    stream=float((torch.cat(res16)-torch.cat(res16).mean(0)).pow(2).sum(1).mean())
    shares={li: float((torch.cat(v)-torch.cat(v).mean(0)).pow(2).sum(1).mean())/stream
            for li,v in mos.items()}
    # per-source transplant -> movement of L16's write along BUS / along RND
    def run_measure(rows, transplant=None):
        cap=[]
        h1=m.transformer.h[16].mlp.register_forward_hook(
            lambda mod,i_,o_: cap.append(o_.detach().reshape(-1,D).float()))
        h2=None
        if transplant is not None:
            s_layer,src_mo=transplant
            h2=m.transformer.h[s_layer].mlp.register_forward_hook(
                lambda mod,i_,o_: src_mo.to(o_.dtype))
        m(rows[:,:-1].contiguous(), rows[:,1:].contiguous())
        h1.remove()
        if h2 is not None: h2.remove()
        return cap[0]
    b16=run_measure(base_rows)
    sig_bus=(b16@BUS).std(0); sig_rnd=(b16@RND).std(0)
    res={}
    for s_layer in range(5,16):
        cap=[]
        h=m.transformer.h[s_layer].mlp.register_forward_hook(
            lambda mod,i_,o_: cap.append(o_.detach()))
        m(src_rows[:,:-1].contiguous(), src_rows[:,1:].contiguous())
        h.remove()
        src_mo=cap[0]
        p16=run_measure(base_rows, transplant=(s_layer,src_mo))
        mv_bus=float((((p16-b16)@BUS)/sig_bus).abs().mean())
        mv_rnd=float((((p16-b16)@RND)/sig_rnd).abs().mean())
        res[s_layer]={'bus':mv_bus,'rnd':mv_rnd,'share':shares[s_layer]}
        print(f'L{s_layer}: bus move {mv_bus:.3f}s | random-span {mv_rnd:.3f}s | '
              f'dilution share {shares[s_layer]:.3f}',flush=True)
    tot_share=sum(shares.values()); tot_bus=sum(r['bus'] for r in res.values())
    ratios={li:(res[li]['bus']/tot_bus)/(shares[li]/tot_share) for li in res}
    mx=max(ratios,key=ratios.get)
    pa=all(r<=2 for r in ratios.values())
    pb=(ratios[mx]>=3) and mx in (14,15)
    out={'per_source':{str(k):v for k,v in res.items()},
         'excess_over_share':{str(k):float(v) for k,v in ratios.items()},
         'max_layer':mx,'max_ratio':float(ratios[mx]),
         'pred_a_diffuse':bool(pa),'pred_b_origin_adjacent':bool(pb)}
    print(f'\nmax excess over dilution share: L{mx} at {ratios[mx]:.2f}x')
    print(f"(a) diffuse (<=2x everywhere): {'HELD' if pa else 'FAILED'}")
    print(f"(b) dedicated adjacent origin (>=3x at L14/15): {'HELD' if pb else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
