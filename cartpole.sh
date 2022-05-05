#!/bin/bash

python train.py \
-env 'CartPole-v1' \
-train_reps 2 \
-eval_reps 2 \
-b 200 \
-r "Discrete" \
-m "IndividualSigma" \
-pat 3 \
-s "CommaSelection" \
-ps 4 \
-os 28 \
-v 1