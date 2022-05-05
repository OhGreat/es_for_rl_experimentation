#!/bin/bash

python train.py \
-env 'Acrobot-v1' \
-exp_reps 2 \
-train_reps 10 \
-eval_reps 20 \
-b 500 \
-r "Discrete" \
-m "IndividualSigma" \
-pat 20 \
-s "PlusSelection" \
-ps 4 \
-os 28 \
-v 1