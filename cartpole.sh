#!/bin/bash

python train.py \
-env 'CartPole-v1' \
-exp_reps 3 \
-train_reps 5 \
-eval_reps 200 \
-b 200 \
-r "Discrete" \
-m "IndividualSigma" \
-pat 3 \
-s "PlusSelection" \
-ps 2 \
-os 10 \
-v 1