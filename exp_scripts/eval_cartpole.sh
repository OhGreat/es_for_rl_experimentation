#!/bin/bash

python eval_model.py \
-model 0 \
-model_weights "model_weights/cartpole_nn_0" \
-env "CartPole-v1" \
-eval_reps 100 \
-render_eval
