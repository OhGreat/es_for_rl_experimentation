#!/bin/bash

python train_network.py \
-env 'Acrobot-v1' \
-exp_reps 3 \
-train_reps 5 \
-eval_reps 150 \
-b 300 \
-r "Discrete" \
-m "IndividualSigma" \
-pat 40 \
-s "CommaSelection" \
-ps 3 \
-os 21 \
-plot_name "Acrobot-v1 nn" \
-v 0
