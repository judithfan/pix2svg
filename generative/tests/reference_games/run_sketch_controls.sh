#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0 python control.py \
    /home/jefan/sketchpad_basic_extract/sketch \
    /home/jefan/sketchpad_basic_extract/sketch_controls/conv_1_1 \
    png 0 --batch_size 32 --cuda

CUDA_VISIBLE_DEVICES=0 python control.py \
    /home/jefan/sketchpad_basic_extract/sketch \
    /home/jefan/sketchpad_basic_extract/sketch_controls/conv_1_2 \
    png 2 --batch_size 32 --cuda

CUDA_VISIBLE_DEVICES=0 python control.py \
    /home/jefan/sketchpad_basic_extract/sketch \
    /home/jefan/sketchpad_basic_extract/sketch_controls/conv_2_1 \
    png 5 --batch_size 32 --cuda

CUDA_VISIBLE_DEVICES=0 python control.py \
    /home/jefan/sketchpad_basic_extract/sketch \
    /home/jefan/sketchpad_basic_extract/sketch_controls/conv_2_2 \
    png 7 --batch_size 32 --cuda

CUDA_VISIBLE_DEVICES=0 python control.py \
    /home/jefan/sketchpad_basic_extract/sketch \
    /home/jefan/sketchpad_basic_extract/sketch_controls/conv_3_1 \
    png 10 --batch_size 32 --cuda

CUDA_VISIBLE_DEVICES=0 python control.py \
    /home/jefan/sketchpad_basic_extract/sketch \
    /home/jefan/sketchpad_basic_extract/sketch_controls/conv_3_2 \
    png 12 --batch_size 32 --cuda

CUDA_VISIBLE_DEVICES=0 python control.py \
    /home/jefan/sketchpad_basic_extract/sketch \
    /home/jefan/sketchpad_basic_extract/sketch_controls/conv_3_3 \
    png 14 --batch_size 32 --cuda

CUDA_VISIBLE_DEVICES=0 python control.py \
    /home/jefan/sketchpad_basic_extract/sketch \
    /home/jefan/sketchpad_basic_extract/sketch_controls/conv_3_4 \
    png 16 --batch_size 32 --cuda

CUDA_VISIBLE_DEVICES=0 python control.py \
    /home/jefan/sketchpad_basic_extract/sketch \
    /home/jefan/sketchpad_basic_extract/sketch_controls/conv_4_1 \
    png 19 --batch_size 32 --cuda

CUDA_VISIBLE_DEVICES=0 python control.py \
    /home/jefan/sketchpad_basic_extract/sketch \
    /home/jefan/sketchpad_basic_extract/sketch_controls/conv_4_2 \
    png 21 --batch_size 32 --cuda

CUDA_VISIBLE_DEVICES=0 python control.py \
    /home/jefan/sketchpad_basic_extract/sketch \
    /home/jefan/sketchpad_basic_extract/sketch_controls/conv_4_3 \
    png 23 --batch_size 32 --cuda

CUDA_VISIBLE_DEVICES=0 python control.py \
    /home/jefan/sketchpad_basic_extract/sketch \
    /home/jefan/sketchpad_basic_extract/sketch_controls/conv_4_4 \
    png 25 --batch_size 32 --cuda

CUDA_VISIBLE_DEVICES=0 python control.py \
    /home/jefan/sketchpad_basic_extract/sketch \
    /home/jefan/sketchpad_basic_extract/sketch_controls/conv_5_1 \
    png 28 --batch_size 32 --cuda

CUDA_VISIBLE_DEVICES=0 python control.py \
    /home/jefan/sketchpad_basic_extract/sketch \
    /home/jefan/sketchpad_basic_extract/sketch_controls/conv_5_2 \
    png 30 --batch_size 32 --cuda

CUDA_VISIBLE_DEVICES=0 python control.py \
    /home/jefan/sketchpad_basic_extract/sketch \
    /home/jefan/sketchpad_basic_extract/sketch_controls/conv_5_3 \
    png 32 --batch_size 32 --cuda

CUDA_VISIBLE_DEVICES=0 python control.py \
    /home/jefan/sketchpad_basic_extract/sketch \
    /home/jefan/sketchpad_basic_extract/sketch_controls/conv_5_4 \
    png 34 --batch_size 32 --cuda

CUDA_VISIBLE_DEVICES=0 python control.py \
    /home/jefan/sketchpad_basic_extract/sketch \
    /home/jefan/sketchpad_basic_extract/sketch_controls/pool_1 \
    png 4 --batch_size 32 --cuda

CUDA_VISIBLE_DEVICES=0 python control.py \
    /home/jefan/sketchpad_basic_extract/sketch \
    /home/jefan/sketchpad_basic_extract/sketch_controls/pool_2 \
    png 9 --batch_size 32 --cuda

CUDA_VISIBLE_DEVICES=0 python control.py \
    /home/jefan/sketchpad_basic_extract/sketch \
    /home/jefan/sketchpad_basic_extract/sketch_controls/pool_3 \
    png 18 --batch_size 32 --cuda

CUDA_VISIBLE_DEVICES=0 python control.py \
    /home/jefan/sketchpad_basic_extract/sketch \
    /home/jefan/sketchpad_basic_extract/sketch_controls/pool_4 \
    png 27 --batch_size 32 --cuda

CUDA_VISIBLE_DEVICES=0 python control.py \
    /home/jefan/sketchpad_basic_extract/sketch \
    /home/jefan/sketchpad_basic_extract/sketch_controls/pool_5 \
    png 36 --batch_size 32 --cuda

CUDA_VISIBLE_DEVICES=0 python control.py \
    /home/jefan/sketchpad_basic_extract/sketch \
    /home/jefan/sketchpad_basic_extract/sketch_controls/fc_0 \
    png 0 --classifier --batch_size 32 --cuda

CUDA_VISIBLE_DEVICES=0 python control.py \
    /home/jefan/sketchpad_basic_extract/sketch \
    /home/jefan/sketchpad_basic_extract/sketch_controls/fc_1 \
    png 3 --classifier --batch_size 32 --cuda
