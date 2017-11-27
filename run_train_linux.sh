#! /bin/bash
python style.py --style-dir './style' \
    --checkpoint-dir './checkpoints/ckpt_03' \
    --test './input/giraffe.jpg' \
    --test-dir './tests/test_003_affine' \
    --content-weight 1.5e1 \
    --checkpoint-iterations 1 \
    --batch-size 20 \
    --epochs 1 \
    --affine \
    --affine-weight 1e4