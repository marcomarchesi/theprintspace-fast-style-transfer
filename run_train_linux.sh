#! /bin/bash

python style.py --style-dir './style' \
    --checkpoint-dir './checkpoints/ckpt_028' \
    --test './input/murray_01.jpg' \
    --test-dir './tests/test_028' \
    --content-weight 1.5e1 \
    --checkpoint-iterations 100 \
    --batch-size 30 \
    --epochs 2 \
    --affine 
