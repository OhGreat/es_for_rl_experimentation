#!/bin/bash

python train.py \
-env 'Acrobot-v1' \
-reps 2 \
-eval_reps 2 \
-b 200 \
-r "Discrete" \
-m "IndividualSigma" \
-pat 10 \
-s "PlusSelection" \
-ps 2 \
-os 4 \
-v 1