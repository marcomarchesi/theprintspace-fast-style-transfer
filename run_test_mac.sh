#! /bin/bash

python evaluate.py --checkpoint '/Users/marcomarchesi/Desktop/fast-style-transfer/checkpoint/checkpoint_bailey_05/fns.ckpt' \
  --in-path '/Users/marcomarchesi/Desktop/fast-style-transfer/input' \
  --out-path '/Users/marcomarchesi/Desktop/fast-style-transfer/output' \
  --allow-different-dimensions