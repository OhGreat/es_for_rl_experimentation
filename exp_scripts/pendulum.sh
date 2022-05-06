#!/bin/bash

python train_network.py \
-exp_name "pendulum_nn_0" \
-model 0 \
-env 'Pendulum-v1' \
-exp_reps 3 \
-train_reps 5 \
-eval_reps 150 \
-b 350 \
-r "Discrete" \
-m "IndividualSigma" \
-pat 40 \
-s "CommaSelection" \
-ps 3 \
-os 21 \
-plot_name "Pendulum nn 0" \
-env_threshold -16.2736044 \
-v 0
