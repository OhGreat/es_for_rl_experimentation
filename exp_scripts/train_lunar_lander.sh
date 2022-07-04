#!/bin/bash

python train_model.py \
-exp_name "lunar_lander_nn_2" \
-model 3 \
-env 'LunarLander-v2' \
-exp_reps 3 \
-train_reps 3 \
-eval_reps 150 \
-b 10000 \
-r "GlobalDiscrete" \
-m "IndividualSigma" \
-pat 5 \
-s "CommaSelection" \
-ps 12 -os 94 \
-plot_name "LunarLander-v2 nn 2" \
-env_threshold 200 \
-v 2

