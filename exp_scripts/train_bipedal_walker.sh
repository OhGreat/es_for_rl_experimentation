#!/bin/bash

python train_network.py \
-exp_name "bipedal_walker_3" \
-model 3 \
-env 'BipedalWalker-v3' \
-exp_reps 3 \
-train_reps 5 \
-eval_reps 150 \
-b 2000 \
-r "Discrete" \
-m "IndividualSigma" \
-pat 40 \
-s "CommaSelection" \
-ps 2 \
-os 14 \
-plot_name "Bipedal Walker nn 3" \
-env_threshold 300 \
-v 2