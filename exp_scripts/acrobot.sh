#!/bin/bash

python train.py \
-env 'Acrobot-v1' \
-exp_reps 3 \
-train_reps 5 \
-eval_reps 150 \
-b 200 \
-r "Discrete" \
-m "IndividualSigma" \
-pat 40 \
-s "CommaSelection" \
-ps 2 \
-os 12 \
-v 1
