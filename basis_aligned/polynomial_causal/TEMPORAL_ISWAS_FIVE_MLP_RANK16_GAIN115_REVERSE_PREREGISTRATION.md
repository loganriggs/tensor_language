# Five-MLP rank-16 gain-1.15 reverse OOD validation — preregistration

Freeze support `MLP0,1,2,3,6`, the old-family all-position rank-16 bases, and gain `1.15` from the
prospective forward curve. Reverse the intervention by starting from donor prompts and installing the
projected base-minus-donor source outputs. Evaluate against the complete native base endpoint. Run the
same operation on donor-side controls and score KL/top-one changes from that native baseline.

Predictions: A hashes, basis availability, native endpoint finiteness, and price pass; B full five-source
reverse patches are functional; C projected gain-1.15 reverse patches have response projection at least
`.8`, RSE at most `.2`, and behavior projection at least `.75` on both tasks; D projected reverse controls
have median KL at most `.02` and zero flips; E forward-versus-reverse behavior projection differs by at
most `.10` for each task. Maximum 28 forwards, no gradients or updates.
