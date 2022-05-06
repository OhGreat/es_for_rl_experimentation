#!/bin/bash

python eval.py \
-model 0 \
-model_weights "model_weights/cartpole_nn_0" \
-env "CartPole-v1" \
-eval_reps 5 \
-render_eval
