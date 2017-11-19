#! /bin/bash
python style.py --style './style/bailey_01.jpg' \
    --checkpoint-dir './checkpoint_bailey_no_affine' \
    --test './input/giraffe.jpg' \
    --test-dir './input/' \
    --content-weight 5e0 \
    --style-weight 1e2 \
    --tv-weight 2e2 \
    --checkpoint-iterations 10 \
    --learning-rate 1.0 \
    --batch-size 20 \
    --epochs 2 \
    --num-examples 1000

python style.py --style './style/bailey_01.jpg' \
    --checkpoint-dir './checkpoint_bailey_affine' \
    --test './input/giraffe.jpg' \
    --test-dir './input/' \
    --content-weight 5e0 \
    --style-weight 1e2 \
    --tv-weight 2e2 \
    --checkpoint-iterations 10 \
    --learning-rate 1.0 \
    --batch-size 20 \
    --epochs 2 \
    --num-examples 1000 \
    --affine