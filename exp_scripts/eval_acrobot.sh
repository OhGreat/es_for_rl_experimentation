#!/bin/bash

python eval.py \
-model 0 \
-model_weights "model_weights/acrobot_nn_0" \
-env "Acrobot-v1" \
-eval_reps 100 \
-render_eval

