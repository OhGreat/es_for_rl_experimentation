#!/bin/bash

python train_network.py \
-exp_name "lunar_lander_nn_0" \
-model 0 \
-env 'LunarLander-v2' \
-exp_reps 5 \
-train_reps 3 \
-eval_reps 150 \
-b 1000 \
-r "Discrete" \
-m "IndividualSigma" \
-pat 40 \
-s "CommaSelection" \
-ps 3 -os 21 \
-plot_name "LunarLander-v2 nn 0" \
-env_threshold 200 \
-v 0

