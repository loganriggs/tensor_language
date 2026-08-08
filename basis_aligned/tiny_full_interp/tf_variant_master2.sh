#!/bin/bash
cd "$(dirname "$0")" || exit 1
while pgrep -f -x "/bin/bash ./tf_variant_master.sh" > /dev/null \
   || pgrep -f -x "/bin/bash ./tf_variant_chain2.sh" > /dev/null \
   || pgrep -f -x "/bin/bash ./tf_variant_train_chain.sh" > /dev/null; do sleep 30; done
./tf_variant_chain3.sh
