#! /bin/bash

python style.py --style-dir './style' \
    --checkpoint-dir './checkpoints/ckpt_015' \
    --test './input/murray_01.jpg' \
    --test-dir './tests/test_015_affine' \
    --content-weight 1.5e1 \
    --checkpoint-iterations 100 \
    --batch-size 20 \
    --epochs 2 \
    --affine \
    --affine-weight 1e0
