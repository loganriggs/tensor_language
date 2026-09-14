# Optional portable shared runtime

Keep the original package execute.py, program.pt and manifest.json unchanged.
Add shared_execute.py containing the exact native-validated SharedParent class
and a filesystem-local loader for the existing program. No imports from the
research repository are permitted in the copied package.

A: class source matches validated implementation byte-for-byte; isolated loading
and execution of both heads replays original<=1e-10relative. B: no additional
serialized tensors; report complete source/package bytes and256KiB extra resident
adapters. Use a new manifest for this optional entry point. Test in an isolated
temporary directory with only original execute.py/program.pt and shared_execute.py.
No new fitting, behavioral claim or overriding old manifest after validation.
