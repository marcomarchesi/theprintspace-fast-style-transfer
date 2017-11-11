#! /bin/bash
python style.py --style '/Users/marcomarchesi/Desktop/fast-style-transfer/style/bailey_01.jpg' \
    --checkpoint-dir '/Users/marcomarchesi/Desktop/fast-style-transfer/checkpoint' \
    --test '/Users/marcomarchesi/Desktop/fast-style-transfer/input/giraffe.jpg' \
    --test-dir '/Users/marcomarchesi/Desktop/fast-style-transfer/input/' \
    --content-weight 1.5e1 \
    --checkpoint-iterations 2 \
    --batch-size 2