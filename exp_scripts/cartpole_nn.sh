#!/bin/bash

python train_network.py \
-exp_name "cartpole_nn_0" \
-model 0 \
-env 'CartPole-v1' \
-exp_reps 1 \
-train_reps 30 \
-eval_reps 150 \
-b 500 \
-r "Discrete" \
-m "IndividualSigma" \
-pat 40 \
-s "CommaSelection" \
-ps 3 \
-os 21 \
-plot_name "CartPole-v1 nn 0" \
-env_threshold 475 \
-v 0
