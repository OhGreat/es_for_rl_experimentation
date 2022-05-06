#!/bin/bash

python train_network.py \
-exp_name "acrobot_nn_0" \
-model 0 \
-env 'Acrobot-v1' \
-exp_reps 5 \
-train_reps 5 \
-eval_reps 150 \
-b 600 \
-r "Discrete" \
-m "IndividualSigma" \
-pat 40 \
-s "CommaSelection" \
-ps 3 \
-os 21 \
-plot_name "Acrobot-v1 nn" \
-env_threshold -100 \
-v 0
