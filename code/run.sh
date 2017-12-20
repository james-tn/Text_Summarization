#!/bin/bash

# first arg is output base directory
[ $# -eq 0 ] && base_dir='../output' || base_dir="$1"

data_dir=$base_dir
out_dir=$data_dir/result

vocab_file=$data_dir/vocab/vocab

dev_tgt_file=$data_dir/abstract/val_000.bin
test_tgt_file=$data_dir/abstract/test_000.bin

dev_src_file=$data_dir/article/val_000.bin
test_src_file=$data_dir/article/test_000.bin

num_train_steps=12000
steps_per_stats=100
vocab_size=60000

mkdir -p "$out_dir"

time python3 -u main.py   \
 --attention=scaled_luong \
 --num_train_steps=$num_train_steps \
 --steps_per_stats=$steps_per_stats \
 --num_layers=3     \
 --num_units=128    \
 --dropout=0.2      \
 --metrics=rouge    \
 --out_dir=$out_dir \
 --vocab_size=$vocab_size \
 --data_dir=$data_dir     \
 --vocab_file=$vocab_file \
 --test_src_file=$test_src_file \
 --test_tgt_file=$test_tgt_file \
 --dev_src_file=$dev_src_file   \
 --dev_tgt_file=$dev_tgt_file   \
 --train_prefix=train_   \
 --dev_prefix=val_       \
 --test_prefix=test_     \
 --beam_width=10



# original example:
#  /usr/bin/python3 -u  /root/text_sum/code/main.py
#    --attention=scaled_luong
#    --num_train_steps=12000
#    --steps_per_stats=100
#    --num_layers=3
#    --num_units=128
#    --dropout=0.2
#    --metrics=rouge
#    --out_dir=../out_dir
#    --vocab_size=60000
#    --data_dir=../data
#    --vocab_file=../data/vocab/vocab
#    --test_src_file=../data/article/test_000.bin
#    --test_tgt_file=../data/abstract/test_000.bin
#    --dev_src_file=../data/article/val_000.bin
#    --dev_tgt_file=../data/abstract/val_000.bin
#    --train_prefix=train_
#    --dev_prefix=val_
#    --test_prefix=test_
#    --beam_width=10
