#!/bin/bash

python train_network.py \
-exp_name "lunar_lander_nn_3" \
-model 3 \
-env 'LunarLander-v2' \
-exp_reps 3 \
-train_reps 5 \
-eval_reps 150 \
-b 2000 \
-r "Intermediate" \
-m "IndividualSigma" \
-pat 20 \
-s "CommaSelection" \
-ps 3 -os 21 \
-plot_name "LunarLander-v2 nn 3" \
-env_threshold 200 \
-v 2

