#! /bin/bash

python evaluate.py --checkpoint '/Users/marcomarchesi/Desktop/fast-style-transfer/checkpoints/scream.ckpt' \
  --in-path '/Users/marcomarchesi/Desktop/fast-style-transfer/input' \
  --out-path '/Users/marcomarchesi/Desktop/fast-style-transfer/output' \
  --allow-different-dimensions