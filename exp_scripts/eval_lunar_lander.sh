#!/bin/bash

python eval.py \
-model 0 \
-model_weights "model_weights/lunar_lander_nn_0" \
-env "LunarLander-v2" \
-eval_reps 100 \
-render_eval

