#!/bin/bash

#CUDA_VISIBLE_DEVICES=3 python train.py /data/wumike/save_states/training_11_28_17/triplet_search/1_1_1/ --photo_augment --match_weight 1.0 --category_weight 1.0 --instance_weight 1.0 --cuda;
CUDA_VISIBLE_DEVICES=3 python train.py /data/wumike/save_states/training_11_28_17/triplet_search/1_1_0/ --photo_augment --match_weight 1.0 --category_weight 1.0 --instance_weight 0.0 --cuda;
# CUDA_VISIBLE_DEVICES=3 python train.py /data/wumike/save_states/training_11_28_17/triplet_search/1_0_1/ --photo_augment --match_weight 1.0 --category_weight 0.0 --instance_weight 1.0 --cuda;
# CUDA_VISIBLE_DEVICES=3 python train.py /data/wumike/save_states/training_11_28_17/triplet_search/0_1_1/ --photo_augment --match_weight 0.0 --category_weight 1.0 --instance_weight 1.0 --cuda;
# CUDA_VISIBLE_DEVICES=3 python train.py /data/wumike/save_states/training_11_28_17/triplet_search/1_0_0/ --photo_augment --match_weight 1.0 --category_weight 0.0 --instance_weight 0.0 --cuda;
# CUDA_VISIBLE_DEVICES=3 python train.py /data/wumike/save_states/training_11_28_17/triplet_search/0_0_1/ --photo_augment --match_weight 0.0 --category_weight 0.0 --instance_weight 1.0 --cuda;
# CUDA_VISIBLE_DEVICES=3 python train.py /data/wumike/save_states/training_11_28_17/triplet_search/0_1_0/ --photo_augment --match_weight 0.0 --category_weight 1.0 --instance_weight 0.0 --cuda;
