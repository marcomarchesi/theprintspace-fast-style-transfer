#! /bin/bash
python style.py --style-dir '/Users/marcomarchesi/Desktop/theprintspace-fast-style-transfer/style' \
    --checkpoint-dir '/Users/marcomarchesi/Desktop/theprintspace-fast-style-transfer/checkpoint/ckpt_007' \
    --test '/Users/marcomarchesi/Desktop/theprintspace-fast-style-transfer/content/portrait_03.jpg' \
    --test-dir '/Users/marcomarchesi/Desktop/theprintspace-fast-style-transfer/content/' \
    --content-weight 1.5e1 \
    --checkpoint-iterations 1 \
    --batch-size 2 \
    --epochs 1 \
    --no-gpu \
<<<<<<< HEAD
    --affine \
    --affine-plus
=======
    --affine 
>>>>>>> 070410bcb0a1ee3b0866bc2f48fb04dfeaa7446c
