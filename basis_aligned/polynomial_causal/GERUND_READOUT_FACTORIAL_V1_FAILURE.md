# V1 publication failure,10 September17:14 UTC

The managed V1 reached the state-artifact save and then failed at
`artifact.stat().st_size()`: st_size is an integer field. No result JSON exists.
Preserve V1 runner, binding, states and log. V2 changes only this call and
its result/binding/state filenames; it reruns the unchanged12-forward science
protocol and binds V1 states for a subsequent CPU replay comparison.
No scientific predictions were read, selected or changed before the repair.
The failed execution is not a valid native receipt.
