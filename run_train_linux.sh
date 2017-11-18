#! /bin/bash
python style.py --style './style/bailey_01.jpg' \
    --checkpoint-dir './checkpoint_bailey' \
    --test './input/giraffe.jpg' \
    --test-dir './input/' \
    --content-weight 1.5e1 \
    --style-weight 1e2 \
    --checkpoint-iterations 10 \
    --batch-size 20 \
    --affine