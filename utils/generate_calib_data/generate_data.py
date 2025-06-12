# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from models.rwkv_v7 import get_mod_rwkv_model, get_rwkv_dataset, get_rwkv_token, eval_lambada
import torch
import json
import sys
import os
import argparse
import torch.nn.functional as F


parser = argparse.ArgumentParser()
parser.add_argument(
    "model",
    type=str,
    help="rwkv model to load",
)


DEV = "cuda:0"
args = parser.parse_args()
model = get_mod_rwkv_model(args.model, rotation=False)
mdoel = model.to(DEV)
model.eval()
tokenizer = get_rwkv_token()

n_vocab = 500 # number of initial tokens for synthesizing data on each GPU.

i_start = 0
if os.path.exists("gen_data/gen.chunk."+str(i_start).zfill(2)+".jsonl"):
    with open("gen_data/gen.chunk."+str(i_start).zfill(2)+".jsonl", "r") as f:
        lines = f.readlines()
        inner_loop = len(lines) % n_vocab
        outer_loop = len(lines) // n_vocab
else:
    inner_loop = 0
    outer_loop = 0

if not os.path.exists("generate_calib_data/gen_data"):
    os.mkdir("generate_calib_data/gen_data")

for j in range(3 + outer_loop, 6):
    for i in range(int(i_start) * n_vocab + inner_loop, (int(i_start)+1) * n_vocab):
        print(i)
        input_ids = torch.tensor([[i]]).cuda()
        print("generating")
        
        outputs1 = model.forward(input_ids)
        probs = F.softmax(outputs1, dim=-1)

        if i < j:
            # Top-1采样：选择概率最高的Token
            next_token = torch.argmax(probs, dim=-1)

        else:
            # 随机采样：根据概率分布采样
            probabilities = torch.softmax(probs, dim=-1)
            next_token = torch.multinomial(probabilities.view(-1), num_samples=1)

        input = next_token[0].tolist() if isinstance(next_token[0].tolist(), list) else [next_token[0].tolist()]
        gen_text = tokenizer.decode(input)
        text_dict = {"text" : gen_text}
        with open("generate_calib_data/gen_data/gen.chunk."+str(i_start).zfill(2)+".jsonl", "a") as f:
            f.write(json.dumps(text_dict))
            f.write('\n')
