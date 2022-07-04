#!/bin/bash

python train_model.py \
-exp_name "bipedal_walker_3" \
-model 3 \
-env 'BipedalWalker-v3' \
-exp_reps 1 \
-train_reps 1 \
-eval_reps 150 \
-b 10000 \
-r "GlobalDiscrete" \
-m "IndividualSigma" \
-pat 5 \
-s "CommaSelection" \
-ps 24 \
-os 144 \
-plot_name "Bipedal Walker nn 3" \
-env_threshold 300 \
-v 2