#!/bin/bash

python train_model.py \
-exp_name "mountain_car_continuous_nn_1" \
-model 1 \
-env 'MountainCarContinuous-v0' \
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
-plot_name "Mountain Car Continuous nn 1" \
-env_threshold 90 \
-v 0