#!/bin/bash

python train_model.py \
-exp_name "bipedal_walker_3" \
-model 3 \
-env 'BipedalWalker-v3' \
-exp_reps 1 \
-train_reps 10 \
-eval_reps 150 \
-b 3000 \
-r "GlobalDiscrete" \
-m "IndividualSigma" \
-pat 10 \
-s "CommaSelection" \
-ps 3 \
-os 21 \
-plot_name "Bipedal Walker nn 3_" \
-env_threshold 300 \
-v 2