#!/usr/bin/env python3
# BQGATE: LIBRARY -- CPU pre-enqueue check (review 25): undefined module-level names + dry-run exit code, for regex-derived runners.
"""Usage: python dod_check_runner.py <runner.py>. Flags Name loads inside functions that are neither local, nor module-level assignments /
imports / defs, nor builtins (the stale-name class that crashed v211 after its 60 forwards and broke v223's dry-run); then runs the
BQLIB_DRYRUN=1 dry-run and reports its exit code. Exit 1 on any finding."""
import ast, builtins, os, subprocess, sys
path = sys.argv[1]; tree = ast.parse(open(path).read())
module_names = set(dir(builtins))
for node in tree.body:
    if isinstance(node, (ast.Import, ast.ImportFrom)): module_names |= {(a.asname or a.name).split(".")[0] for a in node.names}
    elif isinstance(node, (ast.FunctionDef, ast.ClassDef)): module_names.add(node.name)
    elif isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
        for t in (node.targets if isinstance(node, ast.Assign) else [node.target]):
            for n in ast.walk(t):
                if isinstance(n, ast.Name): module_names.add(n.id)
findings = []
def locals_of(fn):
    local = {a.arg for a in fn.args.args + fn.args.kwonlyargs + fn.args.posonlyargs} | ({fn.args.vararg.arg} if fn.args.vararg else set()) | ({fn.args.kwarg.arg} if fn.args.kwarg else set())
    for n in ast.walk(fn):
        if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)): local.add(n.id)
        elif isinstance(n, (ast.FunctionDef, ast.ClassDef)): local.add(n.name)
        elif isinstance(n, (ast.Import, ast.ImportFrom)): local |= {(a.asname or a.name).split(".")[0] for a in n.names}
        elif isinstance(n, ast.arg): local.add(n.arg)
        elif isinstance(n, ast.comprehension):
            for t in ast.walk(n.target):
                if isinstance(t, ast.Name): local.add(t.id)
        elif isinstance(n, ast.ExceptHandler) and n.name: local.add(n.name)
    return local
def check(fn, enclosing):
    local = locals_of(fn) | enclosing        # closures see the enclosing functions' locals
    for n in ast.walk(fn):
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and n.id not in local and n.id not in module_names:
            findings.append(f"{path}:{n.lineno}: undefined name '{n.id}' in {fn.name}()")
    for inner in [n for n in ast.walk(fn) if isinstance(n, ast.FunctionDef) and n is not fn]:
        check(inner, local)
for fn in [n for n in tree.body if isinstance(n, ast.FunctionDef)]:
    check(fn, set())
try:
    for f in sorted(set(findings)): print(f)
except BrokenPipeError: pass
env = dict(os.environ, BQLIB_DRYRUN="1"); r = subprocess.run([sys.executable, path], env=env, capture_output=True, text=True)
print("dry-run exit", r.returncode, "" if r.returncode == 0 else r.stderr.strip().splitlines()[-1] if r.stderr.strip() else "")
sys.exit(1 if findings or r.returncode else 0)
