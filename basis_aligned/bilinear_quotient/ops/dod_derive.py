#!/usr/bin/env python3
# BQGATE: LIBRARY -- AST-aware runner derivation (review 30): rewrite named module constants, swap the docstring, rename receipt stems.
"""Usage: python dod_derive.py <src.py> <dst.py> [--set NAME=<literal>]... [--docstring <file>] [--gate "<pred names>"] [--import OLD=NEW]...
Each --set rewrites the module-level assignment whose target is NAME (single-target assignments only; tuple-unpacking lines are rewritten
by rebuilding the whole line when NAME is one of its targets and every target is given). The receipt stem and candidate id are renamed
from the source and destination file names (run_<stem>.py). Fails loudly if a NAME is not assigned at module level."""
import ast, re, sys
args = sys.argv[1:]; src, dst = args[0], args[1]; sets, imports, doc, gate = {}, [], None, None
i = 2
while i < len(args):
    if args[i] == "--set": k, v = args[i + 1].split("=", 1); sets[k] = v; i += 2
    elif args[i] == "--docstring": doc = open(args[i + 1]).read(); i += 2
    elif args[i] == "--gate": gate = args[i + 1]; i += 2
    elif args[i] == "--import": imports.append(args[i + 1].split("=", 1)); i += 2
    else: raise SystemExit(f"unknown arg {args[i]}")
text = open(src).read(); lines = text.splitlines(keepends=True); tree = ast.parse(text)
done = set()
for node in tree.body:
    if isinstance(node, ast.Assign) and len(node.targets) == 1:
        t = node.targets[0]; names = [t.id] if isinstance(t, ast.Name) else [e.id for e in t.elts] if isinstance(t, ast.Tuple) else []
        hit = [n for n in names if n in sets]
        if not hit: continue
        if len(names) == 1: new = f"{names[0]} = {sets[names[0]]}\n"
        else:
            if not all(n in sets for n in names): raise SystemExit(f"tuple assignment {names}: every target must be given with --set")
            new = ", ".join(names) + " = " + ", ".join(sets[n] for n in names) + "\n"
        lines[node.lineno - 1:node.end_lineno] = [new]; done |= set(hit)
        text = "".join(lines); lines = text.splitlines(keepends=True); tree = ast.parse(text)
missing = set(sets) - done
if missing: raise SystemExit(f"names not assigned at module level: {sorted(missing)}")
text = "".join(lines)
sstem = re.sub(r"^run_|\.py$", "", src.split("/")[-1]); dstem = re.sub(r"^run_|\.py$", "", dst.split("/")[-1])
text = text.replace(sstem, dstem)
for old, new in imports: text = text.replace(old, new)
if doc is not None:
    m = re.search(r'"""[\s\S]*?"""', text); text = text[:m.start()] + doc.strip() + text[m.end():]
if gate is not None: text = re.sub(r"# BQGATE: EXPERIMENT .*", "# BQGATE: EXPERIMENT " + gate, text, count=1)
open(dst, "w").write(text); print("derived", dst, "set:", sorted(done), "stem:", sstem, "->", dstem)
