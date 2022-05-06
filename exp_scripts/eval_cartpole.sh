#!/bin/bash

python eval.py \
-model 1 \
-model_weights "model_weights/cartpole_nn_1" \
-env "CartPole-v1" \
-eval_reps 100 \
