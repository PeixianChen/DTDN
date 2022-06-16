<<<<<<< HEAD
CUDA_VISIVBLE_DEVICES=0,1,2,3 \
python -u baseline_office.py \
--data-dir /home/data/office31/ \
-s webcam \
-t amazon \
-a resnet50 \
-b 88 \
--height 256 \
--width 128 \
--logs-dir ./office31 \
--epoch 19 \
--workers=4 \
--lr 0.01    \
--features 512 \
# --resume \
# --evaluate \
# 'amazon', 'dslr', 'webcam'





=======
CUDA_VISIBLE_DEVICES=2,3,1 \
python -u baseline_office.py \
--data-dir /home/Dataset/office31/ \
-t amazon \
-s dslr \
-a resnet50 \
-b 88 \
--height 224 \
--width 224 \
--logs-dir ./office31 \
--epoch 40 \
--workers=4 \
--lr 0.01    \
--features 2048 \
# 'amazon', 'dslr', 'webcam'

>>>>>>> f0906cafd587b9f863e29ed0904c7c6f81d0db32
