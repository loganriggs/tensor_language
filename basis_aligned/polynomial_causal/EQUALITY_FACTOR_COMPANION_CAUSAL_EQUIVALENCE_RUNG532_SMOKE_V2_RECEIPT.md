# Rung 532 managed v2 smoke receipt

**Completed:** 2026-09-03 12:31 UTC

The separately named v2 smoke completed with exit code 0 and passed its explicit fail-closed instrument predicate.
No CE, circuit-family, task, composition, or OOD outcome was opened.

- corrected core SHA-256: `2207288b731f69a5b540ab101d3b293d2f5ff7831f347d8e59ac00bf7e59e9e2`
- v2 wrapper SHA-256: `53f44a18a0289b79928ac12960cef30a0f93528c084b27949d52389898fb33fa`
- v2 log SHA-256: `53004ae5b1e3cad2e3e2ed25b6b6091a1f93599e7c5a3593825f4ecfc84520a2`

Checks:

- exactly 21 forwards;
- native versus analytical replay logit maximum difference `0`;
- captured native factor product versus parent product maximum difference `0`;
- parent factor reconstruction error at most `4.56e-14`;
- minimum donor-removal edit RMS `7.890`;
- minimum target-arm edit RMS `2.147`;
- zero intended edits with zero RMS;
- peak allocated GPU memory `3,159,521,280` bytes.

The frozen full 2,625-forward run is eligible for the managed queue.
