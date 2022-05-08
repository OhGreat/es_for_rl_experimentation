#!/bin/bash

python eval.py \
-model 3 \
-model_weights "model_weights/bipedal_walker_3" \
-env "BipedalWalker-v3" \
-eval_reps 100 \
-render_eval