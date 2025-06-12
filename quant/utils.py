import numpy as np

# import libpysal
# from libpysal.weights import Queen
# from esda.moran import Moran
# from libpysal.weights import lat2W
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import torch
from typing import Optional
import random


def seed_everything(seed: Optional[int] = None) -> int:
    """
    设置所有随机种子以确保可重复性。

    参数:
    seed (Optional[int]): 随机种子。默认为 None。

    返回:
    int: 使用的随机种子。
    """
    if seed is None:
        seed = random.randint(0, 2**32 - 1)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # 确保在使用 CUDA 时的可重复性
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    return seed


def get_moran_stat(matrix, w=None):
    n, m = matrix.shape
    N = n * m

    # 创建空间权重矩阵
    w = lat2W(n, m, rook=False)  # rook=False for queen contiguity

    # 将权重矩阵转换为行标准化
    w.transform = "r"

    # 计算Moran's I
    moran = Moran(matrix.flatten(), w)
    return moran.I


def draw_img(data, save_path="/data01/home/xuchen/gptvq/experiment/a_imgs/", name="test.png"):
    fig, ax = plt.subplots()

    # 使用seaborn的heatmap函数绘制热图
    # cmap参数可以设置颜色映射，例如"Blues"表示从浅蓝到深蓝
    sns.heatmap(data, cmap="Blues", ax=ax, annot=False, fmt=".1f")

    # 使用matplotlib的contourf函数绘制等高线图
    # levels参数可以设置等高线的数量或具体值
    contour = ax.contourf(data, levels=10, cmap="Blues")

    # 添加颜色条
    # cbar = fig.colorbar(contour, ax=ax)
    # cbar.set_label('Value')

    # 显示图形
    output_path = os.path.join(save_path, name)
    plt.savefig(output_path)
