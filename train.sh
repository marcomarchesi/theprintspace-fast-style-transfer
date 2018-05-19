python style.py --style-image './styles/bailey_04_foreground.jpg' \
    --checkpoint-dir './checkpoints/ckpt_064' \
    --test './inputs/kk_portrait.jpg' \
    --test-dir './tests/test_064' \
    --checkpoint-iterations 1000 \
    --batch-size 30 \
    --epochs 3 \
    --luma