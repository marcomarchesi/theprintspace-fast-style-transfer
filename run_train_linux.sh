#! /bin/bash
python style.py --style './style/bailey_01.jpg' \
    --checkpoint-dir './checkpoint_bailey_07' \
    --test './input/giraffe.jpg' \
    --test-dir './input/' \
    --content-weight 1.5e1 \
    --checkpoint-iterations 100 \
    --batch-size 20 \
    --affine False