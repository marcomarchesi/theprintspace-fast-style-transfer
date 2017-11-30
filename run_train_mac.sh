#! /bin/bash
python style.py --style-dir '/Users/marcomarchesi/Desktop/fast-style-transfer/style' \
    --checkpoint-dir '/Users/marcomarchesi/Desktop/fast-style-transfer/checkpoint/ckpt_03' \
    --test '/Users/marcomarchesi/Desktop/fast-style-transfer/input/giraffe.jpg' \
    --test-dir '/Users/marcomarchesi/Desktop/fast-style-transfer/test/' \
    --content-weight 1.5e1 \
    --checkpoint-iterations 1 \
    --batch-size 2 \
    --epochs 1 \
    --no-gpu