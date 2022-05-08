#!/bin/bash

python eval.py \
-model 3 \
-model_weights "model_weights/lunar_lander_nn_3" \
-env "LunarLander-v2" \
-eval_reps 100 \
-render_eval

