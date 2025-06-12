# Code taken from https://github.com/IST-DASLab/gptq
# Copyright 2022 IST-DASLab, Licensed under the Apache License, Version 2.0
# License is provided for attribution purposes only, Not a Contribution


import torch
import torch.nn as nn
from models.rwkv_v7 import Cus_Mul
from models.rwkv_v6 import Cus_Mul6
import os

DEV = torch.device("cuda:0")


def get_task_name(args):
    if "168M" in args.model:
        name = "rawkv7_0.1B"
    elif "421M" in args.model:
        name = "rawkv7_0.5B"
    elif "1.47B" in args.model:
        name = "rawkv7_1.47B"
    elif "1B" in args.model:
        name = "rawkv6_1B"
    elif "3B" in args.model:
        name = "rawkv6_3B"
    elif "7B" in args.model:
        name = "rawkv6_7B"
    elif "14B" in args.model:
        name = "rawkv6_14B"

    if args.mask_q:
        name_mask = "_mask_q"
    else:
        name_mask = ""

    if args.coherent_type:
        name_coher = f"_{args.coherent_type}"
    else:
        name_coher = ""

    if args.use_vq and args.use_kmeans:
        name = name + "_kmeans" + name_coher + f"_{args.quant_mul}_{args.quant_conv}_{args.groupsize}"
    elif args.use_vq and not args.use_incoherent:
        name = name + "_vp" + name_coher + f"_{args.quant_mul}_{args.quant_conv}_{args.groupsize}"
    elif args.use_incoherent:
        name = name + "_mix" + name_coher + f"_{args.quant_mul}_{args.quant_conv}_{args.groupsize}"
    else:
        if args.use_awq:
            name = name + "_awq" + name_coher + f"_{args.quant_mul}_{args.quant_conv}_{args.groupsize}"
        elif args.use_rtn:
            name = name + "_rtn" + name_mask + f"_{args.quant_mul}_{args.quant_conv}_{args.groupsize}"
        elif args.use_qurot:
            name = name + "_qurot" + name_coher + f"_{args.quant_mul}_{args.quant_conv}_{args.groupsize}"
        elif args.use_vptq:
            name = name + "_vptq" + name_coher + f"_{args.quant_mul}_{args.quant_conv}_{args.groupsize}"
        else:
            name = name + "_gptq" + name_coher + f"_{args.quant_mul}_{args.quant_conv}_{args.groupsize}"

    path = f"experiment/latest/{name}"
    os.makedirs(path, exist_ok=True)
    return name, path


def replace_module_by_name(model, name, new_module):
    """
    通过字符串形式的名字递归访问模型中的子模块或属性。

    Args:
        model: 要访问的模型或模块（如 model.blocks）。
        name (str): 模块的名字字符串，支持多级访问（如 'att.r_k_mul'）。

    Returns:
        目标模块或属性。
    """
    parts = name.split(".")  # 按 "." 分割名字
    module = model
    for part in parts[:-1]:  # 递归到倒数第二级模块
        module = getattr(module, part)

    # 使用 setattr 替换最后一级模块
    setattr(module, parts[-1], new_module)


def find_layers(
    module,
    layers=[nn.Conv2d, nn.Linear],
    name="",
    find_mul=False,
    find_conv=True,
):
    if not find_conv:
        layers = []

    if find_mul:
        layers = layers + [Cus_Mul, Cus_Mul6]  #

    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(child, layers=layers, name=name + "." + name1 if name != "" else name1))
    return res


def find_layers_mul(module, layers=[Cus_Mul, Cus_Mul6], name=""):  #
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(child, layers=layers, name=name + "." + name1 if name != "" else name1))
    return res
