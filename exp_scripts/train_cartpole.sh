#!/bin/bash

python train_model.py \
-exp_name "cartpole_nn_1" \
-model 1 \
-env 'CartPole-v1' \
-exp_reps 3 \
-train_reps 5 \
-eval_reps 150 \
-b 500 \
-r "Discrete" \
-m "IndividualSigma" \
-pat 40 \
-s "CommaSelection" \
-ps 3 \
-os 21 \
-plot_name "CartPole-v1 nn 1" \
-env_threshold 475 \
-v 0
