#!/bin/bash

python eval.py \
-model 1 \
-model_weights "model_weights/acrobot_nn_1" \
-env "Acrobot-v1" \
-eval_reps 100 \
-render_eval

