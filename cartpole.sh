#!/bin/bash

python train_network.py \
-env 'CartPole-v1' \
-exp_reps 10 \
-train_reps 6 \
-eval_reps 150 \
-b 150 \
-r "Discrete" \
-m "IndividualSigma" \
-pat 5 \
-s "CommaSelection" \
-ps 4 \
-os 28 \
-v 1
