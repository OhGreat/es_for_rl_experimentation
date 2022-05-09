#!/bin/bash

python train_model.py \
-exp_name "lunar_lander_nn_3" \
-model 3 \
-env 'LunarLander-v2' \
-exp_reps 3 \
-train_reps 8 \
-eval_reps 150 \
-b 3000 \
-r "Discrete" \
-m "IndividualSigma" \
-pat 10 \
-s "CommaSelection" \
-ps 3 -os 18 \
-plot_name "LunarLander-v2 nn 3" \
-env_threshold 200 \
-v 2

