#!/bin/bash

set -x

MODEL_PATH=Qwen/Qwen2.5-1.5B-Instruct  # replace it with your local file path

python3 -m verl.trainer.main \
    config=examples/config_agg_math_1p5b_fullresp_8gpu.yaml \
    worker.actor.model.model_path=${MODEL_PATH}


