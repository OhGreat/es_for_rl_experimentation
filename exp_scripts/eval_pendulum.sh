#!/bin/bash

python eval_model.py \
-model 0 \
-model_weights "model_weights/pendulum_nn_0" \
-env "Pendulum-v1" \
-eval_reps 100 \
-render_eval
