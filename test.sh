# python test_tta.py \
#     --model-name efficientnetb7 \
#     --test-batch-size 2048 \
#     --image-size 224 \
#     --gpu 4,5,6,7 \
#     --weights results/efficientnetb7_balance/best_f1_checkpoint.pth.tar \
#     --mode val \
#     --out output/val

python test.py \
    --model-name efficientnetb7 \
    --test-batch-size 128 \
    --image-size 224 \
    --gpu 0 \
    --weights results/efficientnetb7_balance/best_f1_checkpoint.pth.tar \
    --mode test \
    --val-csv test_data/test_images.csv \
    --out output/test
