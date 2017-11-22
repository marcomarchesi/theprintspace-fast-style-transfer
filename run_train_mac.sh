#! /bin/bash
python style.py --style-dir '/Users/marcomarchesi/Desktop/fast-style-transfer/style' \
    --checkpoint-dir '/Users/marcomarchesi/Desktop/fast-style-transfer/checkpoint' \
    --test '/Users/marcomarchesi/Desktop/fast-style-transfer/input/giraffe.jpg' \
    --test-dir '/Users/marcomarchesi/Desktop/fast-style-transfer/test/' \
    --content-weight 1.5e1 \
    --checkpoint-iterations 2 \
    --batch-size 5 \
    --epochs 2 \
    --no-gpu \
    --num-examples 10 \
    --affine 