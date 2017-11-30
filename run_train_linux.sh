#! /bin/bash
python style.py --style-dir './style' \
    --checkpoint-dir './checkpoints/ckpt_006' \
    --test './input/giraffe.jpg' \
    --test-dir './tests/test_006' \
    --content-weight 1.5e1 \
    --checkpoint-iterations 100 \
    --batch-size 20 \
    --epochs 2