#!/usr/bin/env python3
"""Pre-queue static gate for BQ experiment scripts. LESSONS 18, 19, 21.

Usage: python3 ops/gate.py <script.py>

LESSONS 21: this gate has produced THREE false positives (regex case-sensitivity,
module-level tuple unpacking, nested defs / lambda args). On FAIL, reproduce the
finding by hand before touching the script -- fix the GATE if the gate is wrong.
"""
import re, ast, sys, builtins


def _targets(n):
    out = set()
    for t in getattr(n, 'targets', []):
        if isinstance(t, ast.Name):
            out.add(t.id)
        elif isinstance(t, ast.Tuple):                      # LESSONS 21 fix 2
            out |= {e.id for e in t.elts if isinstance(e, ast.Name)}
    return out


def _args(fn):
    a = fn.args
    got = {x.arg for x in list(a.args) + list(a.posonlyargs) + list(a.kwonlyargs)}
    if a.vararg:
        got.add(a.vararg.arg)
    if a.kwarg:
        got.add(a.kwarg.arg)
    return got


def _bound(scope, fn=None):
    loc = _args(fn) if fn else set()
    for n in ast.walk(scope):
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
            loc.add(n.id)
        if isinstance(n, ast.Assign):
            loc |= _targets(n)
        if isinstance(n, (ast.Import, ast.ImportFrom)):
            loc |= {a.asname or a.name.split('.')[0] for a in n.names}
        if isinstance(n, ast.FunctionDef) and n is not scope:  # LESSONS 21 fix 3a
            loc.add(n.name); loc |= _args(n)
        if isinstance(n, ast.Lambda):                          # LESSONS 21 fix 3b
            loc |= _args(n)
        if isinstance(n, (ast.For, ast.comprehension)):
            t = n.target
            if isinstance(t, ast.Name):
                loc.add(t.id)
            elif isinstance(t, ast.Tuple):
                loc |= {e.id for e in t.elts if isinstance(e, ast.Name)}
        # `except E as e:` binds a plain str, not a Name/Store node, so the walk above never saw it and
        # every `except ... as e: print(e)` read as an undefined name. Found 2026-08-29 on ops/bqlib.py.
        # Binding it scope-wide would HIDE the one real bug in this shape -- Python DELETES the name at
        # the end of the handler, so a use AFTER the handler is a genuine NameError -- so that case gets
        # its own check below rather than being swallowed here.
        if isinstance(n, ast.ExceptHandler) and n.name:
            loc.add(n.name)
        if isinstance(n, (ast.Global, ast.Nonlocal)):
            loc |= set(n.names)
    return loc


def _except_name_escapes(tree):
    """`except E as e:` names are deleted at the end of the handler; using one after it is a NameError."""
    bad = []
    for h in [n for n in ast.walk(tree) if isinstance(n, ast.ExceptHandler) and n.name]:
        inside = {id(x) for x in ast.walk(h)}
        for fn in [n for n in ast.walk(tree)
                   if isinstance(n, (ast.FunctionDef, ast.Module, ast.AsyncFunctionDef))]:
            for u in ast.walk(fn):
                if (isinstance(u, ast.Name) and u.id == h.name
                        and isinstance(u.ctx, ast.Load) and id(u) not in inside):
                    bad.append((h.name, u.lineno))
    return sorted(set(bad))


def _nocomment(src):
    """source with whole-line comments dropped -- three checks fired on their own
    explanatory comments before this existed."""
    return '\n'.join(l for l in src.split('\n') if not l.lstrip().startswith('#'))


def gate(path):
    s = open(path).read()
    fails = []
    try:
        tree = ast.parse(s)
    except SyntaxError as e:
        return [f'SYNTAX ERROR: {e}']

    mod = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
    for n in tree.body:
        if isinstance(n, ast.Assign):
            mod |= _targets(n)
        if isinstance(n, (ast.Import, ast.ImportFrom)):
            mod |= {a.asname or a.name.split('.')[0] for a in n.names}

    for fn in [f for f in tree.body if isinstance(f, ast.FunctionDef)]:
        used = {n.id for n in ast.walk(fn)
                if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
        u = sorted(used - _bound(fn, fn) - mod - set(dir(builtins)))
        if u:
            fails.append(f'{fn.name}(): possibly undefined {u}')
    # MODULE-LEVEL undefined names. The undefined-name check above walks FUNCTION bodies only, so a
    # name used at module scope and never bound was invisible. 2026-08-29: a fork left `ALPHA` in the
    # result payload after the block defining it had been replaced -- legal syntax, gate PASS, and a
    # NameError that would have fired AFTER the whole run, while building the argument to report().
    # Discovered by hand-auditing a script the gate had just passed, which is not a strategy.
    # module dunders are injected by the interpreter, not bound by any statement. Without them
    # this check flagged 226 of 227 scripts on `__file__` alone -- LESSON 67's shape exactly,
    # caught because the first thing it was run against was the whole corpus and not one file.
    MODDUNDERS = {'__file__', '__name__', '__doc__', '__package__', '__spec__', '__loader__',
                  '__builtins__', '__debug__', '__path__'}
    modnames = _bound(tree) | MODDUNDERS
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        for u in ast.walk(n):
            if isinstance(u, ast.Name) and isinstance(u.ctx, ast.Load) \
                    and u.id not in modnames and u.id not in dir(builtins):
                fails.append(f'line {u.lineno}: module-level `{u.id}` is used but never bound '
                             f'-- a NameError at run time, usually a fork that dropped its definition')

    for nm, ln in _except_name_escapes(tree):
        fails.append(f'line {ln}: `{nm}` is an except-handler name used OUTSIDE its handler -- Python deletes it there (NameError)')

    # A NESTED FUNCTION'S FREE VARIABLE MUST NOT BE ASSIGNED LATER IN ITS ENCLOSING FUNCTION.
    # §1815's ce_dominance_check named a Pareto marker `m` inside main(); that made `m` a local of
    # main, shadowing the module-level model that the nested build() closes over, and every arm died
    # with "cannot access free variable 'm'". The existing undefined-name check cannot see it: `m` IS
    # assigned in main, just after the nested call runs.
    for fn in [f for f in tree.body if isinstance(f, ast.FunctionDef)]:
        outer_assigned = set()
        for n in ast.walk(fn):
            if isinstance(n, ast.Assign):
                outer_assigned |= _targets(n)
            elif isinstance(n, ast.For):
                outer_assigned |= _targets(n)
        nested = [n for n in ast.walk(fn)
                  if isinstance(n, ast.FunctionDef) and n is not fn]
        for inner in nested:
            free = {n.id for n in ast.walk(inner)
                    if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
            free -= _bound(inner, inner)
            shadowed = sorted((free & outer_assigned & mod) - set(dir(builtins)))
            if shadowed:
                fails.append(
                    f'{fn.name}(): nested {inner.name}() reads module global(s) {shadowed} '
                    f'that {fn.name}() also assigns -- the assignment shadows the global for the '
                    f'whole function and the nested call will fail')

    # A MODULE-LEVEL CONSTANT ASSIGNED TWICE is almost always a merge error in this lineage, where
    # scripts are built by editing a previous one. §1822's bottom_up_gain_rescue set DEPTHS = (0,3,5)
    # and an inherited DEPTHS = (-1,7,10,13) eight lines later silently won; the run executed the
    # wrong depths for three minutes and died on a KeyError for an arm that was never created.
    seen_const, dupes = set(), set()
    for n in tree.body:
        if isinstance(n, ast.Assign):
            for t in _targets(n):
                if t.isupper() and t in seen_const:
                    dupes.add(t)
                seen_const.add(t)
    if dupes:
        fails.append(f'module-level constant(s) assigned twice: {sorted(dupes)} '
                     f'-- the later assignment silently wins')

    # every function used AS A VALUE must return something (LESSONS 18).
    # A call that is the whole of an Expr statement is statement-use, not value-use.
    stmt_calls = {id(n.value) for n in ast.walk(tree) if isinstance(n, ast.Expr)
                  and isinstance(n.value, ast.Call)}
    value_used = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) \
                and id(n) not in stmt_calls:
            value_used.add(n.func.id)
    for fn in [f for f in tree.body if isinstance(f, ast.FunctionDef)]:
        if fn.name in value_used and fn.name != 'main':
            if not any(isinstance(n, ast.Return) and n.value is not None
                       for n in ast.walk(fn)):
                fails.append(f'{fn.name}(): called as a value but has no `return <value>`')

    # AT LEAST three DISTINCT registered predictions (LESSONS 19).
    # LESSONS 21 fix 1: [A-Za-z] in the key pattern.
    # S1700 fix: the rule was `== 3`, which rejected a legitimate FOURTH prediction. Codex's
    # pre-execution hardening of whole_model_heldout added a pred_d carrying bootstrap
    # intervals, and my own amended whole_model_v1_floor does the same. Three was a floor
    # against under-registering, never a ceiling; enforcing it as a ceiling penalised adding
    # MORE falsifiable content. Distinctness and the a/b/c core are still required.
    keys = re.findall(r"'(pred_[A-Za-z0-9_]+)':", s)
    letters = {k.split("_")[1] for k in keys}
    # 2026-08-29: ops/bqlib.py introduced a THIRD file class -- a shared library that no experiment
    # result comes out of. The predicate rules exist to stop an EXPERIMENT shipping unregistered; they
    # are meaningless for a module and were rejecting it. The exemption is deliberately hard to trip by
    # accident: it needs an explicit marker AND no result-JSON write AND no module-level main() call.
    # It skips ONLY the three predicate checks -- every other check still runs on the library -- and it
    # announces itself, so it can never fire silently on something that should have been registered.
    lib_marker = bool(re.search(r'^# BQGATE: LIBRARY\b', s, re.M))
    writes_results = bool(re.search(r'json\.dump\s*\(', s))
    calls_main = bool(re.search(r'^main\(\)\s*$', s, re.M)) or "__main__" in s
    is_library = lib_marker and not writes_results and not calls_main
    if is_library:
        print(f'GATE: {path} is a LIBRARY (marker + no result write + no main()); '
              f'predicate checks skipped, all other checks applied')
    else:
        if lib_marker:
            fails.append('carries the LIBRARY marker but writes results or calls main() -- '
                         'it is an experiment and must register predicates')
        if len(keys) < 3:
            fails.append(f'expected at least 3 pred_* keys, found {len(keys)}: {keys}')
        if len(letters) != len(keys):
            fails.append(f'pred keys not distinct: {keys}')
        if not {'a', 'b', 'c'} <= letters:
            fails.append(f'pred keys must include a, b and c: {sorted(letters)}')

    # site consistency (LESSONS 20: forward extent must match the component set)
    st = re.search(r'SITE_STOP = (\d+)', s)
    up = re.search(r'SITE_UP = (\d+)', s)
    sites = set(re.findall(r"'site': (\d+)", s))
    if st and up and st.group(1) != up.group(1):
        fails.append(f'SITE_STOP={st.group(1)} != SITE_UP={up.group(1)}')
    if st and sites and sites != {st.group(1)}:
        fails.append(f'cell sites {sites} != SITE_STOP {st.group(1)}')

    # a module-level name bound to an EMPTY literal, iterated inside the results
    # dict, silently writes an empty record even though the run scored correctly
    empty = set()
    for n in tree.body:
        if isinstance(n, ast.Assign) and isinstance(n.value, (ast.List, ast.Dict, ast.Set)):
            vals = getattr(n.value, 'elts', None)
            if vals is None:
                vals = getattr(n.value, 'keys', [])
            if not vals:
                empty |= _targets(n)
    # ...unless the code MUTATES it. `ITER_DELTA = []` filled by ITER_DELTA.append() inside a nested
    # build() is a live accumulator, not a dead literal. Only these four verbs rescue it; a name that
    # is merely READ stays flagged, which is the case the check was written for.
    empty -= {m2 for m2 in empty
              if re.search(rf'\b{re.escape(m2)}\.(append|extend|update|add)\(', _nocomment(s))}
    for fn in [f for f in tree.body if isinstance(f, ast.FunctionDef)]:
        for n in ast.walk(fn):
            if isinstance(n, ast.comprehension) and isinstance(n.iter, ast.Name) \
                    and n.iter.id in empty:
                fails.append(f'{fn.name}(): comprehension iterates `{n.iter.id}`, which is '
                             f'an EMPTY module-level literal -- the record will serialise empty')


    # LESSONS 56: the runtime BANNER names a different experiment than the file.
    # Built by editing a predecessor, the docstring gets rewritten and the print does not -- so the
    # log header, which is the first thing a reader sees and the thing a write-up is quoted from,
    # attributes the numbers to the wrong run. Measured: 16 of 101 ops scripts with a house-convention
    # banner named a DIFFERENT experiment, all 16 true positives.
    _stop = {'ops', 'py', 'the', 'a', 'is', 'it', 'of', 'and', 'in', 'to', 'v2',
             'check', 'scan', 'probe', 'test'}

    def _toks(t):
        return {w for w in re.split(r'[^a-z0-9]+', t.lower()) if w and w not in _stop}
    for fn in [f for f in ast.walk(tree) if isinstance(f, ast.FunctionDef) and f.name == 'main']:
        for n in ast.walk(fn):
            if not (isinstance(n, ast.Call) and getattr(n.func, 'id', '') == 'print' and n.args):
                continue
            v = n.args[0]
            lit = (v.values[0].value if isinstance(v, ast.JoinedStr) and v.values
                   and isinstance(v.values[0], ast.Constant) and isinstance(v.values[0].value, str)
                   else (v.value if isinstance(v, ast.Constant) and isinstance(v.value, str)
                         else None))
            if not lit or ' | ' not in lit:
                continue
            head = lit.split(' | ')[0].strip()
            hw = head.split()
            if len(hw) < 2 or not all(w.isupper() or not w.isalpha() for w in hw[:2]):
                continue
            stem = __import__('os').path.basename(path)[:-3]
            if not (_toks(stem) & _toks(head)):
                fails.append(f'main(): the log banner says {head!r} but the file is '
                             f'{stem!r} -- they share no word. A banner carried over from the '
                             f'script this one was edited from mis-attributes the run in the log')
            break


    # LESSONS 59: the tail indexes an arm label suffix that the arm loop never produces.
    # A predecessor called run_g twice per arm with '_raw'/'_seq' labels; a successor that calls it
    # once with a bare label keeps the tail's f'{name}_seq' keys and dies at the reporting step AFTER
    # every arm has run. The GPU work is done and thrown away, so this is worth a static check.
    labels = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Call) and getattr(n.func, 'id', '') == 'run_g' and n.args:
            v = n.args[0]
            if isinstance(v, ast.Constant) and isinstance(v.value, str):
                labels.add(v.value)
            elif isinstance(v, ast.JoinedStr):
                tailpc = v.values[-1]
                if isinstance(tailpc, ast.Constant) and isinstance(tailpc.value, str):
                    labels.add('*' + tailpc.value)
    if labels:
        for suf in ('_raw', '_seq', '_global', '_matched'):
            produced = any(l.endswith(suf) for l in labels)
            wanted = re.search(rf"\{{[A-Za-z_0-9\[\]'\"]+\}}{suf}'", s) is not None
            if wanted and not produced:
                fails.append(f"results are indexed with the suffix '{suf}' but no run_g() call "
                             f"produces a label ending in it (labels seen: {sorted(labels)}) -- "
                             f"the reporting step will KeyError after every arm has run")


    # LESSONS 63: the pred_* KEY and the pred_* DOCSTRING line disagree, and the docstring line is
    # inherited VERBATIM from another ops script -- i.e. the registered questions were rewritten in one
    # place and not the other. Measured: 14 such lines across the tree, in both directions (stale
    # docstring with fresh code, and fresh docstring with stale code). Restricted to VERBATIM-inherited
    # lines because a short paraphrase that merely shares no stem is common and benign.
    _pstop = {'the', 'and', 'is', 'it', 'of', 'a', 'at', 'in', 'to', 'by', 'on', 'its', 'be', 'for',
              'than', 'not', 'this', 'that', 'all', 'any', 'no', 'are', 'was', 'if', 'true', 'false',
              'pred', 'with', 'from'}

    def _ptok(t):
        return {w for w in re.split(r'[^a-z0-9]+', t.lower())
                if w and w not in _pstop and len(w) > 2}
    _doc = {m.group(1): m.group(2).strip()
            for m in re.finditer(r'#\s+pred_([a-d])\s+(.{20,120})', s)}
    _key = dict(re.findall(r"'pred_([a-d])_([a-z0-9_]+)':", s))
    _here = __import__('os').path.abspath(path)
    _dir = __import__('os').path.dirname(_here)
    for _L in 'abc':
        if _L not in _doc or _L not in _key:
            continue
        if _ptok(_doc[_L]) & _ptok(_key[_L]):
            continue
        _from = []
        for _q in __import__('glob').glob(_dir + '/*.py'):
            if __import__('os').path.abspath(_q) == _here:
                continue
            try:
                if _doc[_L][:60] in open(_q).read():
                    _from.append(__import__('os').path.basename(_q))
            except Exception:
                pass
        if _from:
            fails.append(f"pred_{_L}: the docstring line {_doc[_L][:50]!r} is inherited VERBATIM "
                         f"from {_from[:2]} but the result key is 'pred_{_L}_{_key[_L]}' -- the "
                         f"registered question was rewritten in one place and not the other")


    # LESSONS 64: indexing curve['full'] when TRANKS contains no None. The rank-sweep lineage keys its
    # results dict by str(rank) with None -> 'full', so a script that drops full rank from TRANKS but
    # keeps an inherited curve['full'] lookup dies in the TAIL, after every build has run. Fourth
    # tail-inheritance failure in this codebase; unlike the others it is statically decidable.
    _tr = re.search(r'^TRANKS\s*=\s*\(([^)]*)\)', s, re.M)
    # strip comment lines first: a docstring that merely MENTIONS the pattern is not a lookup, and
    # this check fired on its own explanatory comment the first time it was used.
    _code = _nocomment(s)
    if _tr and 'None' not in _tr.group(1) and re.search(r"curve\['full'\]", _code):
        fails.append("TRANKS contains no None (so the results dict has no 'full' key) but the code "
                     "indexes curve['full'] -- this KeyErrors in the reporting block, after every "
                     "build has already run")


    # LESSONS 64 (generalised): int() applied to a LADDER element when LADDER holds non-numeric names.
    # The rank-sweep lineage prices storage with int(r); a successor whose ladder holds ARM NAMES or
    # 'full' dies in the TAIL after every build. Sixth tail-inheritance failure here, and the second
    # time this exact line did it, so the check is widened from the 'full' special case.
    # search CODE only: this check fired on its own explanatory comment the first time, exactly as the
    # curve['full'] one did. `_code` is defined above.
    _ld = re.search(r'^LADDER\s*=\s*(.+)$', _code, re.M)
    if _ld and re.search(r'\bint\(\s*[rR]\s*\)', _code):
        _src = _ld.group(1)
        # LADDER is often built from another constant (LADDER = list(ARMS2)); resolve one hop, or the
        # check reads no literals and goes SILENT -- which is exactly how it missed the sixth instance.
        for _nm in re.findall(r'\b([A-Z][A-Z0-9_]{2,})\b', _src):
            _o = re.search(rf'^{_nm}\s*=\s*(.+)$', s, re.M)
            if _o:
                _src += ' ' + _o.group(1)
        _lits = re.findall(r"'([^']*)'", _src)
        _bad = [x for x in _lits if not x.isdigit()]
        if _bad:
            fails.append(f'LADDER holds non-numeric entries {_bad} but the code calls int(r) on a '
                         f'ladder element -- this ValueErrors in the reporting block, after every '
                         f'build has already run')

    # call-arity consistency for the helpers whose return shape varies
    for helper in ('abs_mass',):
        n_ret = [len(n.value.elts) for f in ast.walk(tree)
                 if isinstance(f, ast.FunctionDef) and f.name == helper
                 for n in ast.walk(f)
                 if isinstance(n, ast.Return) and isinstance(n.value, ast.Tuple)]
        if n_ret:
            want = n_ret[0]
            for u in re.findall(rf'(\S[^=\n]*)=\s*{helper}\(', s):
                if len(u.split(',')) != want:
                    fails.append(f'{helper}() returns {want} values; unpack `{u.strip()}` '
                                 f'takes {len(u.split(","))}')

    # `del X` inside a loop whose body never assigns X. String surgery that leaves trailing
    # indentation silently re-indents the FOLLOWING retained line into the block just inserted:
    # stream_input_closure.py absorbed `del tables, Ecov, Eunc, A` into the for-loop above it, parsed
    # clean, passed this gate, and died 271s in on the loop's second iteration. Delete-in-loop of a
    # loop-invariant name is the signature, and it is always a bug.
    for node in ast.walk(tree):
        if not isinstance(node, (ast.For, ast.While)):
            continue
        # the loop's OWN target rebinds every iteration, so `for name, sc in ...: ... del sc` is
        # legal -- that idiom is what the second draft flagged in rank_crossover{,_v2}, both exit=0.
        bound = {t.id for t in ast.walk(node.target) if isinstance(t, ast.Name)} \
            if isinstance(node, ast.For) else set()
        for b in node.body:
            for n2 in ast.walk(b):
                if isinstance(n2, ast.Assign):
                    # targets may be tuples: `bank, seen, n = build(...)` binds all three. Matching
                    # only bare ast.Name flagged 28 of 156 working scripts on the first draft.
                    bound |= {t.id for tg in n2.targets for t in ast.walk(tg)
                              if isinstance(t, ast.Name)}
                elif isinstance(n2, (ast.AugAssign, ast.AnnAssign)) and isinstance(n2.target, ast.Name):
                    bound.add(n2.target.id)
                elif isinstance(n2, (ast.For, ast.comprehension)):
                    bound |= {t.id for t in ast.walk(n2.target) if isinstance(t, ast.Name)}
                elif isinstance(n2, (ast.With, ast.AsyncWith)):
                    bound |= {t.id for it in n2.items if it.optional_vars
                              for t in ast.walk(it.optional_vars) if isinstance(t, ast.Name)}
        for b in node.body:
            for n2 in ast.walk(b):
                if isinstance(n2, ast.Delete):
                    for tgt in n2.targets:
                        if isinstance(tgt, ast.Name) and tgt.id not in bound:
                            fails.append(f'line {n2.lineno}: `del {tgt.id}` sits inside a loop that '
                                         f'never assigns {tgt.id} -- UnboundLocalError on the second '
                                         f'iteration (an indentation splice, per stream_input_closure)')

    # a result key whose words do not appear in ITS OWN registered pred_X line. §1902 shipped
    # `pred_a_restored_not_self_consistent_program_concentrated_on_frequent_targets` -- a correct prefix
    # with an ancestor's question welded to the tail, because a rename substituted the front and left the
    # rest. Measured 45 of 708 keys across 177 result files. The LESSON 63 check does not see it: that one
    # compares the DOCSTRING against the key it found, and the front matched.
    _dl = {}
    for _m in re.finditer(r'^#\s+pred_([a-d])\s+(.*?)(?=^#\s+pred_[a-d]\s|^[^#])',
                          s, re.M | re.S):
        _dl[_m.group(1)] = _m.group(2).lower()
    _STOP = {'the', 'and', 'not', 'for', 'its', 'his', 'her', 'that', 'this', 'with', 'from',
             'than', 'more', 'less', 'all', 'any', 'per', 'via', 'pct', 'vs'}
    for _m in re.finditer(r"'pred_([a-d])_([a-z0-9_]+)'", _nocomment(s)):
        _p, _body = _m.group(1), _m.group(2)
        if _p not in _dl:
            continue
        # the WELDING signature is length AND low overlap together. Low overlap alone fires on short
        # coherent names that merely use different vocabulary from the docstring -- measured 3 such
        # (settled_ridge_scan 'pred_c_known_answer_kept_slice' and two others), all true names of their
        # own question. Every genuine welded key measured was over 42 chars after the prefix.
        _w = [w for w in _body.split('_') if len(w) > 3 and w not in _STOP]
        if len(_w) < 4 or len(_body) <= 42:
            continue
        _hit = sum(1 for w in _w if w in _dl[_p])
        if _hit / len(_w) < 0.4:
            fails.append(f"result key 'pred_{_p}_{_body}' shares only {_hit}/{len(_w)} content words "
                         f"with its own registered pred_{_p} line, and is {len(_body)} chars -- a rename "
                         f"likely left an ancestor's question welded to the tail (§1902)")
    return fails



if __name__ == '__main__':
    f = gate(sys.argv[1])
    print('\n'.join(f) if f else 'no findings')
    print('GATE:', 'FAIL' if f else 'PASS')
    sys.exit(1 if f else 0)
