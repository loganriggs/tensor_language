#!/bin/bash
# Wait for chain 1 (exact-name pgrep on the script's own basename -- substring
# pgrep self-matches, which cost the parent program two runs), then chain 2.
cd "$(dirname "$0")" || exit 1
while pgrep -f -x "/bin/bash ./tf_variant_train_chain.sh" > /dev/null; do sleep 20; done
./tf_variant_chain2.sh
