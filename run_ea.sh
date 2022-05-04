#!/bin/bash

python train.py \
-env 'CartPole-v1' \
-reps 10 \
-eval_reps 10 \
-b 300 \
-r "Discrete" \
-m "IndividualSigma" \
-pat 3 \
-s "CommaSelection" \
-ps 4 \
-os 28 \
-v 1