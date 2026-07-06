"""Reverse-engineer HOW the trained 4-attention-layer model does hop-3 (three chained lookups).
For a hop-3 query [Q, e, H3, a] with a=f^3(e), extract each attention layer's pattern at the
answer position and see which positions it reads. Hypothesis: the layers COMPOSE the lookups —
successive layers move the "current entity" one hop along the chain (e -> f(e) -> f^2(e) -> f^3(e))
by attending to the relevant binding pair. This is the zoom-into-a-circuit payoff that the real-LM
featurizer work could not deliver (localized computation in a toy model).

Run: python hop_circuit.py
"""

import torch

from deep_model import DeepModel, SPECS
from hop_data import sample_docs, E, Q, H0, N_CTX, V, ANS_OFFSET
from model import Attention

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
D_MODEL, N_HEAD = 128, 4


def load(name="attn4-rms-seed0"):
    spec = SPECS[name.split("-")[0]]
    m = DeepModel(V, D_MODEL, N_HEAD, spec, N_CTX, norm="rms",
                  attention="bilinear", residual="lerp", mlp_residual="add").to(DEVICE)
    m.load_state_dict(torch.load(f"runs_hop/{name}/model.pt", map_location=DEVICE))
    return m.eval()


def fmap_from_bindings(tokens):
    """bindings are tokens[0:2E] as [e, f(e)] pairs -> reconstruct f."""
    f = {}
    for i in range(E):
        e, fe = int(tokens[2 * i]), int(tokens[2 * i + 1])
        f[e] = fe
    return f


def layer_patterns(model, tokens):
    """Per attention layer: pattern (n_head, seq, seq), applying layers in order."""
    x = model.embed(tokens)
    pats = []
    for layer in model.layers:
        if isinstance(layer, Attention):
            pats.append(layer.pattern(x)[0].detach())          # (n_head, seq, seq)
        x = layer(x)
    return pats


def run():
    g = torch.Generator().manual_seed(7)
    tokens, qa, qk = sample_docs(1, g)
    tok = tokens[0]
    f = fmap_from_bindings(tok)
    # binding-pair position of entity value v (the pair [v, f(v)] -> position of v)
    vpos = {int(tok[2 * i]): 2 * i for i in range(E)}
    # find a hop-3 query block
    blk = next(j for j in range(len(qk[0])) if int(qk[0][j]) == 3)
    base = 2 * E + 4 * blk
    e = int(tok[base + 1]); ans_pos = base + ANS_OFFSET
    chain = [e, f[e], f[f[e]], f[f[f[e]]]]                      # e, f e, f^2 e, f^3 e (=answer)
    print(f"hop-3 query: e={e}  chain e->f->f2->f3 = {chain}  (answer f^3(e)={chain[3]})", flush=True)
    print(f"binding positions: e@{vpos[chain[0]]}, f(e)@{vpos[chain[1]]}, f2(e)@{vpos[chain[2]]}, f3(e)@{vpos[chain[3]]}", flush=True)

    m = load()
    pats = layer_patterns(m, tok.unsqueeze(0).to(DEVICE))
    print(f"\n{len(pats)} attention layers. Attention FROM the answer position ({ans_pos}); "
          f"top-3 attended positions per head, and which chain-entity's binding they sit at:", flush=True)
    posmap = {vpos[chain[k]]: f"f^{k}(e)={chain[k]}" for k in range(4)}
    posmap[vpos[chain[k]] + 1 if False else base + 1] = "query-e"
    for L, p in enumerate(pats):
        row = p[:, ans_pos, :].mean(0)                          # avg over heads, attention from ans_pos
        top = torch.topk(row, 4).indices.tolist()
        labeled = []
        for t in top:
            lab = posmap.get(t, "")
            # also label if it's the binding VALUE slot (odd pos) of a chain entity
            if not lab:
                for k in range(4):
                    if t == vpos[chain[k]] + 1:
                        lab = f"val f^{k+1}(e)"
                labeled.append(f"pos{t}{'('+lab+')' if lab else ''}")
            else:
                labeled.append(f"pos{t}({lab})")
        print(f"  layer {L}: attends -> {labeled}", flush=True)
    print("\nif successive layers attend to successive chain-entities' bindings, the model "
          "COMPOSES the lookups layer-by-layer (chained-retrieval circuit reverse-engineered)", flush=True)


if __name__ == "__main__":
    run()
