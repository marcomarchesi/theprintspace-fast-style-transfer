#! /bin/bash

python evaluate.py --checkpoint './checkpoint_bailey_07/fns.ckpt' \
  --in-path './input' \
  --out-path './output' \
  --allow-different-dimensions