#! /bin/bash
python style.py --style-dir '/Users/marcomarchesi/Desktop/theprintspace-fast-style-transfer/style' \
    --checkpoint-dir '/Users/marcomarchesi/Desktop/theprintspace-fast-style-transfer/checkpoint/ckpt_007' \
    --test '/Users/marcomarchesi/Desktop/theprintspace-fast-style-transfer/content/giraffe.jpg' \
    --test-dir '/Users/marcomarchesi/Desktop/theprintspace-fast-style-transfer/content/' \
    --content-weight 1.5e1 \
    --checkpoint-iterations 1 \
    --batch-size 30 \
    --epochs 1 \
    --no-gpu \
    --affine \
    --affine-weight 5e2

