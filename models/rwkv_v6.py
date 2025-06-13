########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import torch, types, os, gc, math, json
import numpy as np
import torch.nn as nn
from torch.nn import functional as F

ctx_len = 4096
head_size_a = 64  # don't change
head_size_divisor = 8  # don't change


def get_args(model_path):
    args = types.SimpleNamespace()

    MODEL_PATH = model_path

    args.D_MIX_LORA = 32  # generate TIME_MIX for w,k,v,r,g
    args.D_DECAY_LORA = 64

    if "1B" in MODEL_PATH:
        args.n_layer = 24
        args.n_embd = 2048

    if "3B" in MODEL_PATH:
        args.n_layer = 32
        args.n_embd = 2560

    if "7B" in MODEL_PATH:
        args.n_layer = 32
        args.n_embd = 4096
        args.D_MIX_LORA = 64
        args.D_DECAY_LORA = 128

    if "14B" in MODEL_PATH:
        args.n_layer = 61
        args.n_embd = 4096
        args.D_MIX_LORA = 64
        args.D_DECAY_LORA = 128

    args.vocab_size = 65536
    args.ctx_len = 4096
    args.head_size_a = 64  # don't change
    args.head_size_divisor = 8  # don't change
    return args


class Cus_Mul6(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.weight = weight

    def forward(self, x1, x2):
        return x1 * x2


########################################################################################################
# CUDA Kernel
########################################################################################################


from torch.utils.cpp_extension import load

wkv6_cuda = load(
    name="wkv6",
    sources=["cuda/wkv6_op.cpp", f"cuda/wkv6_cuda.cu"],
    verbose=True,
    extra_cuda_cflags=["-res-usage", "--use_fast_math", "-O3", "-Xptxas -O3", "--extra-device-vectorization", f"-D_N_={head_size_a}", f"-D_T_={ctx_len}"],
)


class WKV_6(torch.autograd.Function):
    @staticmethod
    def forward(ctx, B, T, C, H, r, k, v, w, u):  # forward: r, k, v, w, u => y
        with torch.no_grad():
            assert r.dtype == torch.bfloat16
            assert k.dtype == torch.bfloat16
            assert v.dtype == torch.bfloat16
            assert w.dtype == torch.bfloat16
            assert u.dtype == torch.bfloat16
            assert head_size_a == C // H
            ctx.B = B
            ctx.T = T
            ctx.C = C
            ctx.H = H
            assert r.is_contiguous()
            assert k.is_contiguous()
            assert v.is_contiguous()
            assert w.is_contiguous()
            assert u.is_contiguous()
            ctx.save_for_backward(r, k, v, w, u)
            y = torch.empty((B, T, C), device=r.device, dtype=torch.bfloat16, memory_format=torch.contiguous_format)  # .uniform_(-100, 100)
            wkv6_cuda.forward(B, T, C, H, r, k, v, w, u, y)
            return y

    @staticmethod
    def backward(ctx, gy):  # backward: gy => gr, gk, gv, gw, gu
        with torch.no_grad():
            assert gy.dtype == torch.bfloat16
            B = ctx.B
            T = ctx.T
            C = ctx.C
            H = ctx.H
            assert gy.is_contiguous()
            r, k, v, w, u = ctx.saved_tensors
            gr = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format)  # .uniform_(-100, 100)
            gk = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format)  # .uniform_(-100, 100)
            gv = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format)  # .uniform_(-100, 100)
            gw = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format)  # .uniform_(-100, 100)
            gu = torch.empty((B, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format)  # .uniform_(-100, 100)
            wkv6_cuda.backward(B, T, C, H, r, k, v, w, u, gy, gr, gk, gv, gw, gu)
            gu = torch.sum(gu, 0).view(H, C // H)
            return (None, None, None, None, gr, gk, gv, gw, gu)  # return gradients for r,k,v,w,u


def RUN_CUDA_RWKV6(B, T, C, H, r, k, v, w, u):
    return WKV_6.apply(B, T, C, H, r, k, v, w, u)


########################################################################################################
# RWKV TimeMix
########################################################################################################


class RWKV_Tmix_x060(nn.Module):
    def __init__(self, args, layer_id):
        super().__init__()
        D_MIX_LORA = args.D_MIX_LORA
        D_DECAY_LORA = args.D_DECAY_LORA

        self.args = args
        self.layer_id = layer_id

        self.head_size = args.head_size_a
        self.n_head = args.dim_att // self.head_size
        assert args.dim_att % self.n_head == 0

        with torch.no_grad():
            ratio_0_to_1 = layer_id / (args.n_layer - 1)  # 0 to 1
            ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, args.n_embd)
            for i in range(args.n_embd):
                ddd[0, 0, i] = i / args.n_embd

            # fancy time_mix
            self.time_maa_x = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_w = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_v = nn.Parameter(1.0 - (torch.pow(ddd, ratio_1_to_almost0) + 0.3 * ratio_0_to_1))
            self.time_maa_r = nn.Parameter(1.0 - torch.pow(ddd, 0.5 * ratio_1_to_almost0))
            self.time_maa_g = nn.Parameter(1.0 - torch.pow(ddd, 0.5 * ratio_1_to_almost0))

            self.time_maa_w1 = nn.Parameter(torch.zeros(args.n_embd, D_MIX_LORA * 5))
            self.time_maa_w2 = nn.Parameter(torch.zeros(5, D_MIX_LORA, args.n_embd).uniform_(-0.01, 0.01))

            # fancy time_decay
            decay_speed = torch.ones(args.dim_att)
            for n in range(args.dim_att):
                decay_speed[n] = -6 + 5 * (n / (args.dim_att - 1)) ** (0.7 + 1.3 * ratio_0_to_1)
            self.time_decay = nn.Parameter(decay_speed.reshape(1, 1, args.dim_att))

            self.time_decay_w1 = nn.Parameter(torch.zeros(args.n_embd, D_DECAY_LORA))
            self.time_decay_w2 = nn.Parameter(torch.zeros(D_DECAY_LORA, args.dim_att).uniform_(-0.01, 0.01))

            tmp = torch.zeros(args.dim_att)
            for n in range(args.dim_att):
                zigzag = ((n + 1) % 3 - 1) * 0.1
                tmp[n] = ratio_0_to_1 * (1 - (n / (args.dim_att - 1))) + zigzag
            self.time_faaaa = nn.Parameter(tmp.reshape(self.n_head, self.head_size))

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.receptance = nn.Linear(args.n_embd, args.dim_att, bias=False)
        self.key = nn.Linear(args.n_embd, args.dim_att, bias=False)
        self.value = nn.Linear(args.n_embd, args.dim_att, bias=False)
        self.output = nn.Linear(args.dim_att, args.n_embd, bias=False)
        self.gate = nn.Linear(args.n_embd, args.dim_att, bias=False)
        self.ln_x = nn.GroupNorm(self.n_head, args.dim_att, eps=(1e-5) * (args.head_size_divisor**2))

    def jit_func(self, x):
        B, T, C = x.size()

        xx = self.time_shift(x) - x

        xxx = x + xx * self.time_maa_x
        xxx = torch.tanh(xxx @ self.time_maa_w1).view(B * T, 5, -1).transpose(0, 1)
        xxx = torch.bmm(xxx, self.time_maa_w2).view(5, B, T, -1)  # [5, 10, 32]   [5, 32, 2048]  [5, 1, 10, 2048]
        mw, mk, mv, mr, mg = xxx.unbind(dim=0)  # [1, 10, 2048]

        xw = x + xx * (self.time_maa_w + mw)
        xk = x + xx * (self.time_maa_k + mk)
        xv = x + xx * (self.time_maa_v + mv)
        xr = x + xx * (self.time_maa_r + mr)
        xg = x + xx * (self.time_maa_g + mg)

        r = self.receptance(xr)
        k = self.key(xk)
        v = self.value(xv)
        g = F.silu(self.gate(xg))

        ww = torch.tanh(xw @ self.time_decay_w1) @ self.time_decay_w2
        w = self.time_decay + ww

        return r, k, v, g, w

    def jit_func_2(self, x, g):
        B, T, C = x.size()
        x = x.view(B * T, C)

        x = self.ln_x(x).view(B, T, C)
        x = self.output(x * g)
        return x

    def forward(self, x):
        B, T, C = x.size()
        H = self.n_head

        r, k, v, g, w = self.jit_func(x)
        x = RUN_CUDA_RWKV6(B, T, C, H, r, k, v, w, u=self.time_faaaa)

        return self.jit_func_2(x, g)


class RWKV_Tmix_x060_Mod(nn.Module):
    def __init__(self, ori_m, args):
        super().__init__()
        D_MIX_LORA = args.D_MIX_LORA
        D_DECAY_LORA = args.D_DECAY_LORA

        self.args = ori_m.args
        self.layer_id = ori_m.layer_id

        self.head_size = ori_m.head_size
        self.n_head = ori_m.n_head

        with torch.no_grad():
            ratio_0_to_1 = ori_m.layer_id / (args.n_layer - 1)  # 0 to 1
            ratio_1_to_almost0 = 1.0 - (ori_m.layer_id / args.n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, args.n_embd)
            for i in range(args.n_embd):
                ddd[0, 0, i] = i / args.n_embd

            self.linear_w2_1 = nn.Linear(D_MIX_LORA, args.n_embd, bias=False)
            self.linear_w2_1.weight.data = ori_m.time_maa_w2[0]
            self.linear_w2_1.weight.data = self.linear_w2_1.weight.data.T

            self.linear_w2_2 = nn.Linear(D_MIX_LORA, args.n_embd, bias=False)
            self.linear_w2_2.weight.data = ori_m.time_maa_w2[1]
            self.linear_w2_2.weight.data = self.linear_w2_2.weight.data.T

            self.linear_w2_3 = nn.Linear(D_MIX_LORA, args.n_embd, bias=False)
            self.linear_w2_3.weight.data = ori_m.time_maa_w2[2]
            self.linear_w2_3.weight.data = self.linear_w2_3.weight.data.T

            self.linear_w2_4 = nn.Linear(D_MIX_LORA, args.n_embd, bias=False)
            self.linear_w2_4.weight.data = ori_m.time_maa_w2[3]
            self.linear_w2_4.weight.data = self.linear_w2_4.weight.data.T

            self.linear_w2_5 = nn.Linear(D_MIX_LORA, args.n_embd, bias=False)
            self.linear_w2_5.weight.data = ori_m.time_maa_w2[4]
            self.linear_w2_5.weight.data = self.linear_w2_5.weight.data.T
            # self.time_maa_w2 = ori_m.time_maa_w2

            # self.time_maa_w1 = nn.Parameter(torch.zeros(args.n_embd, D_MIX_LORA*5))
            self.linear_w1 = nn.Linear(args.n_embd, D_MIX_LORA * 5, bias=False)
            self.linear_w1.weight = ori_m.time_maa_w1
            self.linear_w1.weight.data = self.linear_w1.weight.data.T

            # fancy time_mix
            self.time_maa_x = ori_m.time_maa_x
            self.time_maa_w = ori_m.time_maa_w
            self.time_maa_k = ori_m.time_maa_k
            self.time_maa_v = ori_m.time_maa_v
            self.time_maa_r = ori_m.time_maa_r
            self.time_maa_g = ori_m.time_maa_g

            self.maa_x_mul = Cus_Mul6(self.time_maa_x)
            self.maa_w_mul = Cus_Mul6(self.time_maa_w)
            self.maa_k_mul = Cus_Mul6(self.time_maa_k)
            self.maa_v_mul = Cus_Mul6(self.time_maa_v)
            self.maa_r_mul = Cus_Mul6(self.time_maa_r)
            self.maa_g_mul = Cus_Mul6(self.time_maa_g)

            # fancy time_decay
            decay_speed = torch.ones(args.dim_att)
            for n in range(args.dim_att):
                decay_speed[n] = -6 + 5 * (n / (args.dim_att - 1)) ** (0.7 + 1.3 * ratio_0_to_1)
            self.time_decay = ori_m.time_decay

            # self.time_decay_w1 = nn.Parameter(torch.zeros(args.n_embd, D_DECAY_LORA))
            # self.time_decay_w2 = nn.Parameter(torch.zeros(D_DECAY_LORA, args.dim_att).uniform_(-0.01, 0.01))

            self.linear_t1 = nn.Linear(args.n_embd, D_MIX_LORA, bias=False)
            self.linear_t1.weight = ori_m.time_decay_w1
            self.linear_t1.weight.data = self.linear_t1.weight.data.T

            self.linear_t2 = nn.Linear(D_DECAY_LORA, args.dim_att, bias=False)
            self.linear_t2.weight = ori_m.time_decay_w2
            self.linear_t2.weight.data = self.linear_t2.weight.data.T

            tmp = torch.zeros(args.dim_att)
            for n in range(args.dim_att):
                zigzag = ((n + 1) % 3 - 1) * 0.1
                tmp[n] = ratio_0_to_1 * (1 - (n / (args.dim_att - 1))) + zigzag
            self.time_faaaa = ori_m.time_faaaa

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.receptance = ori_m.receptance
        self.key = ori_m.key
        self.value = ori_m.value
        self.output = ori_m.output
        self.gate = ori_m.gate
        self.ln_x = ori_m.ln_x

    def jit_func(self, x):
        B, T, C = x.size()

        xx = self.time_shift(x) - x

        # xxx = x + xx * self.time_maa_x
        xxx = x + self.maa_x_mul(xx, self.time_maa_x)

        # xxx = torch.tanh(xxx @ self.time_maa_w1).view(B*T, 5, -1).transpose(0, 1)
        xxx = torch.tanh(self.linear_w1(xxx)).view(B * T, 5, -1).transpose(0, 1)

        # xxx = torch.bmm(xxx, self.time_maa_w2).view(5, B, T, -1)
        # xxx = self.linear_w2(xxx).view(5, B, T, -1)
        mw = self.linear_w2_1(xxx[0]).unsqueeze(0)
        mk = self.linear_w2_2(xxx[1]).unsqueeze(0)
        mv = self.linear_w2_3(xxx[2]).unsqueeze(0)
        mr = self.linear_w2_4(xxx[3]).unsqueeze(0)
        mg = self.linear_w2_5(xxx[4]).unsqueeze(0)

        # mw, mk, mv, mr, mg = xxx.unbind(dim=0)

        xw = x + xx * (self.time_maa_w + mw)
        xk = x + xx * (self.time_maa_k + mk)
        xv = x + xx * (self.time_maa_v + mv)
        xr = x + xx * (self.time_maa_r + mr)
        xg = x + xx * (self.time_maa_g + mg)

        xw = x + self.maa_w_mul(xx, (self.time_maa_w + mw))
        xk = x + self.maa_k_mul(xx, (self.time_maa_k + mk))
        xv = x + self.maa_v_mul(xx, (self.time_maa_v + mv))
        xr = x + self.maa_r_mul(xx, (self.time_maa_r + mr))
        xg = x + self.maa_g_mul(xx, (self.time_maa_g + mg))

        r = self.receptance(xr)
        k = self.key(xk)
        v = self.value(xv)
        g = F.silu(self.gate(xg))

        ww = self.linear_t2(torch.tanh(self.linear_t1(xw)))
        w = self.time_decay + ww

        return r, k, v, g, w

    def jit_func_2(self, x, g):
        B, T, C = x.size()
        x = x.view(B * T, C)

        x = self.ln_x(x).view(B, T, C)
        x = self.output(x * g)
        return x

    def forward(self, x):
        B, T, C = x.size()
        H = self.n_head

        r, k, v, g, w = self.jit_func(x)
        x = RUN_CUDA_RWKV6(B, T, C, H, r, k, v, w, u=self.time_faaaa)

        return self.jit_func_2(x, g)


########################################################################################################
# RWKV ChannelMix
########################################################################################################


class RWKV_CMix_x060(nn.Module):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        with torch.no_grad():  # fancy init of time_mix
            ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, args.n_embd)
            for i in range(args.n_embd):
                ddd[0, 0, i] = i / args.n_embd
            self.time_maa_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_r = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))

            self.maa_k_mul = Cus_Mul6(self.time_maa_k)
            self.maa_r_mul = Cus_Mul6(self.time_maa_r)

        self.key = nn.Linear(args.n_embd, args.dim_ffn, bias=False)
        self.receptance = nn.Linear(args.n_embd, args.n_embd, bias=False)
        self.value = nn.Linear(args.dim_ffn, args.n_embd, bias=False)

    def forward(self, x):
        xx = self.time_shift(x) - x
        # xk = x + xx * self.time_maa_k
        # xr = x + xx * self.time_maa_r
        xk = x + self.maa_k_mul(xx, self.time_maa_k)
        xr = x + self.maa_r_mul(xx, self.time_maa_r)

        k = self.key(xk)
        k = torch.relu(k) ** 2
        kv = self.value(k)
        return torch.sigmoid(self.receptance(xr)) * kv


########################################################################################################
# RWKV Block
########################################################################################################


class Block(nn.Module):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id

        self.ln1 = nn.LayerNorm(args.n_embd)
        self.ln2 = nn.LayerNorm(args.n_embd)

        if self.layer_id == 0:
            self.ln0 = nn.LayerNorm(args.n_embd)

        self.att = RWKV_Tmix_x060(args, layer_id)
        self.ffn = RWKV_CMix_x060(args, layer_id)

    def forward(self, x):

        if self.layer_id == 0:
            x = self.ln0(x)

        x = x + self.att(self.ln1(x))
        x = x + self.ffn(self.ln2(x))

        return x


########################################################################################################
# RWKV Model
########################################################################################################


class RWKV(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        args.dim_att = args.n_embd
        args.dim_ffn = int((args.n_embd * 3.5) // 32 * 32)

        assert args.n_embd % 32 == 0
        assert args.dim_att % 32 == 0
        assert args.dim_ffn % 32 == 0

        self.emb = nn.Embedding(args.vocab_size, args.n_embd)

        self.blocks = nn.ModuleList([Block(args, i) for i in range(args.n_layer)])

        self.ln_out = nn.LayerNorm(args.n_embd)
        self.head = nn.Linear(args.n_embd, args.vocab_size, bias=False)

        # self.init_params() # !!! When you train RWKV from scratch, try my initialization for best performance !!!

    def forward(self, idx):

        x = self.emb(idx)

        for block in self.blocks:
            x = block(x)

        x = self.ln_out(x)
        x = self.head(x)

        return x

    def init_params(self):
        m = self.state_dict()
        n_params = 0

        for n in self.state_dict():
            p = m[n]
            shape = p.shape

            s0 = str(shape[0]) if len(shape) > 0 else ""
            s1 = str(shape[1]) if len(shape) > 1 else ""
            s2 = str(shape[2]) if len(shape) > 2 else ""
            print(f"{s0.ljust(5)} {s1.ljust(5)} {s2.ljust(5)} {n}", end="")

            scale = 1.0
            if "ln_" in n or ".ln" in n or "time_" in n or n.endswith("_w") or n.endswith("_w1") or n.endswith("_w2") or n.endswith("_bias"):
                if "ln_x.weight" in n:
                    layer_scale = (1 + int(n.split(".")[1])) / self.args.n_layer
                    m[n] = (p * 0.0) + (layer_scale**0.7)
                else:
                    m[n] = p
                print()
            elif n == "emb.weight":
                m[n] = p
                scale = -1e-4
                nn.init.uniform_(
                    m[n], a=scale, b=-scale
                )  # !!! If you are using positional embedding, maybe it's better to remove block.0.ln0, and use default initialization for emb.weight instead of my uniform_(a=-1e-4, b=1e-4) !!!
                print(f" [scale {scale}]")
            elif n == "head.weight":
                m[n] = p
                if self.args.vocab_size > self.args.n_embd:
                    scale = 0.5 * math.sqrt(self.args.vocab_size / self.args.n_embd)
                else:
                    scale = 0.5
                nn.init.orthogonal_(m[n], gain=scale)
                print(f" [scale {scale}]")
            else:
                assert n.endswith(".weight")  # should always be true

                for kk in [".att.output.", ".ffn.value.", ".ffn.receptance."]:
                    if kk in n:
                        scale = 0
                for kk in [".att.key."]:
                    if kk in n:
                        scale = 0.1
                for kk in [".att.gate."]:
                    if kk in n:
                        scale = 0.1

                print(f" [scale {scale}]")

                m[n] = torch.empty((shape[0], shape[1]), device=p.device)
                if scale == 0:
                    nn.init.zeros_(m[n])
                else:
                    nn.init.orthogonal_(m[n], gain=scale)

            n_params += m[n].numel()

        print("model params", n_params)
        gc.collect()
        torch.cuda.empty_cache()


########################################################################################################
# RWKV Tokenizer (slow version)
########################################################################################################


class RWKV_TOKENIZER:
    # table: list[list[list[bytes]]]
    # good: list[set[int]]
    # wlen: list[int]
    def __init__(self, file_name):
        self.idx2token = {}
        sorted = []  # must be already sorted
        lines = open(file_name, "r", encoding="utf-8").readlines()
        for l in lines:
            idx = int(l[: l.index(" ")])
            x = eval(l[l.index(" ") : l.rindex(" ")])
            x = x.encode("utf-8") if isinstance(x, str) else x
            assert isinstance(x, bytes)
            assert len(x) == int(l[l.rindex(" ") :])
            sorted += [x]
            self.idx2token[idx] = x

        self.token2idx = {}
        for k, v in self.idx2token.items():
            self.token2idx[v] = int(k)

        # precompute some tables for fast matching
        self.table = [[[] for j in range(256)] for i in range(256)]
        self.good = [set() for i in range(256)]
        self.wlen = [0 for i in range(256)]

        for i in reversed(range(len(sorted))):  # reverse order - match longer tokens first
            s = sorted[i]
            if len(s) >= 2:
                s0 = int(s[0])
                s1 = int(s[1])
                self.table[s0][s1] += [s]
                self.wlen[s0] = max(self.wlen[s0], len(s))
                self.good[s0].add(s1)

    def encodeBytes(self, src: bytes):  # -> list[int]:
        src_len: int = len(src)
        tokens: list[int] = []
        i: int = 0
        while i < src_len:
            s: bytes = src[i : i + 1]

            if i < src_len - 1:
                s1: int = int(src[i + 1])
                s0: int = int(src[i])
                if s1 in self.good[s0]:
                    sss: bytes = src[i : i + self.wlen[s0]]
                    try:
                        s = next(filter(sss.startswith, self.table[s0][s1]))
                    except:
                        pass
            tokens.append(self.token2idx[s])
            i += len(s)

        return tokens

    def decodeBytes(self, tokens):
        return b"".join(map(lambda i: self.idx2token[i], tokens))

    def encode(self, src: str):
        return self.encodeBytes(src.encode("utf-8"))

    def decode(self, tokens):
        return self.decodeBytes(tokens).decode("utf-8")

    def printTokens(self, tokens):
        for i in tokens:
            s = self.idx2token[i]
            try:
                s = s.decode("utf-8")
            except:
                pass
            print(f"{repr(s)}{i}", end=" ")
            # print(repr(s), i)
        print()


tokenizer = RWKV_TOKENIZER("ckpt/v6-Finch-1B6-HF/rwkv_vocab_v20230424.txt")


########################################################################################################
# RWKV utils
########################################################################################################
def get_mod_rwkv_model(model_path=None):
    args = get_args(model_path)
    model_params = torch.load(model_path, map_location="cpu")
    model = RWKV(args).bfloat16().cuda()
    model.load_state_dict(model_params, strict=False)

    for index, block in enumerate(model.blocks):
        model.blocks[index].att = RWKV_Tmix_x060_Mod(block.att, args)
    return model


def get_rwkv_dataset(n_samples=-1):
    with open(f"models/lambada_test.jsonl", "r", encoding="utf-8") as f:
        todo = [json.loads(line) for line in f]
        todo = [[doc["text"].rsplit(" ", 1)[0], " " + doc["text"].rsplit(" ", 1)[1]] for doc in todo]
    if n_samples > 0:
        return todo[:n_samples]
    return todo


def get_rwkv_txt_dataset(n_samples=-1):
    lines_list = []

    # 使用with语句打开文件，确保文件会被正确关闭
    with open("/data01/home/xuchen/gptvq/data/LAMBADA/sample/18096.txt", "r", encoding="utf-8") as file:
        # 逐行读取文件
        for id, line in enumerate(file):
            # 去除每行末尾的换行符并添加到列表中
            if id < n_samples:
                lines_list.append(line.strip())
            else:
                break

    return lines_list


def get_rwkv_token():
    tokenizer = RWKV_TOKENIZER("ckpt/v6-Finch-1B6-HF/rwkv_vocab_v20230424.txt")
    return tokenizer


def eval_lambada(dataset, model, tokenizer, logger):
    print("\nCheck LAMBADA...")
    xsum = 0
    xcnt = 0
    xacc = 0
    for d in dataset:
        src = [0] + tokenizer.encode(d[0])
        dst = tokenizer.encode(d[1])

        logits = 0
        correct = True
        out = model.forward(torch.tensor(src + dst).reshape(1, -1).cuda())
        for i in range(len(dst)):
            ooo = out[0, len(src) - 1 + i].float()
            probs = F.softmax(ooo, dim=-1)
            logits += math.log(probs[dst[i]])
            if torch.argmax(probs).item() != dst[i]:
                correct = False

        xcnt += 1
        xsum += logits
        xacc += 1 if correct else 0
        if xcnt % 100 == 0 or xcnt == len(dataset):
            logger.info(f"{xcnt}, 'ppl', {round(math.exp(-xsum / xcnt), 2)}, 'acc', {round(xacc/xcnt*100, 2)}")


import datasets, torch, torch.nn as nn, tqdm, random, transformers
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm
import os


def eval_ppl_(model, test_loader, tokenizer, seqlen=-1, limit=-1, logger=None):
    nlls = []
    # nsamples = test_loader.numel() // seqlen
    xsum = 0
    xcnt = 0
    xacc = 0

    for d in tqdm(test_loader):
        if len(d["text"]) < 30:
            continue
        d["text"] = d["text"].rstrip(" \n").rstrip(" .")
        data = [d["text"].rsplit(" ", 1)[0], " " + d["text"].rsplit(" ", 1)[1]]
        src = [0] + tokenizer.encode(data[0])
        dst = tokenizer.encode(data[1])

        logits = 0
        correct = True
        out = model.forward(torch.tensor(src + dst).reshape(1, -1).cuda())
        for i in range(len(dst)):
            ooo = out[0, len(src) - 1 + i].float()
            probs = F.softmax(ooo, dim=-1)
            logits += math.log(probs[dst[i]])
            if torch.argmax(probs).item() != dst[i]:
                correct = False

        xcnt += 1
        xsum += logits
        xacc += 1 if correct else 0
        if xcnt % 100 == 0 or xcnt == len(test_loader):
            logger.info(f"{xcnt}, 'ppl', {round(math.exp(-xsum / xcnt), 2)}, 'acc', {round(xacc/xcnt*100, 2)}")

    #     batch = test_loader[:, (i * seqlen) : ((i + 1) * seqlen)].to(model.device)
    #     net_name = model.name.lower() if hasattr(model,"name") else type(model).__name__.lower()
    #     if "opt" in net_name:
    #         outputs = model.model.model.decoder(batch)
    #         hidden_states = outputs[0]
    #         logits = model.model.lm_head(hidden_states)
    #     elif "llama" in net_name or "mixtral" in net_name:
    #         #import pdb;pdb.set_trace()
    #         outputs = model(batch)
    #         logits = outputs['logits'];outputs = None
    #     elif "falcon" in net_name:
    #         outputs = model.model.transformer(batch)
    #         hidden_states = outputs[0]
    #         logits = model.model.lm_head(hidden_states)
    #     elif "glm" in net_name:
    #         outputs = model(batch)
    #         logits = outputs['logits'];outputs = None
    #     shift_logits = logits[:, :-1, :]
    #     shift_labels = test_loader[:, (i * seqlen) : ((i + 1) * seqlen)][
    #         :, 1:
    #     ].to(logits.device)
    #     loss_fct = nn.CrossEntropyLoss()
    #     loss = loss_fct(
    #         shift_logits.view(-1, shift_logits.size(-1)),
    #         shift_labels.view(-1),
    #     )
    #     neg_log_likelihood = loss.float() * seqlen
    #     nlls.append(neg_log_likelihood)
    #     if i == limit:
    #         break
    # ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * seqlen))
    # return ppl.item()


def eval_ppl(model, tokenizer, seqlen=2048, limit=-1, logger=None):
    model.eval()
    # wiki_testdata = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    # # wiki_testloader = tokenizer("\n\n".join(wiki_testdata["text"]), return_tensors="pt")
    # wiki_ppl = eval_ppl_(model, wiki_testdata, tokenizer, seqlen, limit,logger)
    # print(f'wiki ppl : {wiki_ppl}')

    # c4_testdata = load_dataset(
    #     'allenai/c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation'
    # )

    c4_testdata = load_dataset("/data01/home/chenzx/.cache/huggingface/datasets/allenai/c4", data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"}, split="validation")

    random.seed(0)
    random_data = []
    for _ in range(5000):
        data_id = random.randint(0, len(c4_testdata) - 1)
        random_data.append(c4_testdata[data_id])

    c4_ppl = eval_ppl_(model, random_data, tokenizer, seqlen, limit, logger)
    print(f"c4 ppl : {c4_ppl}")


########################################################################################################
# RWKV Inference
########################################################################################################
#  Paris [probability 94.11%]
#  France [probability 0.62%]
#  the [probability 0.58%]
#  pari [probability 0.48%]
#  Se [probability 0.14%]

# use https://huggingface.co/BlinkDL/rwkv-6-world/blob/main/RWKV-x060-World-1B6-v2.1-20240328-ctx4096.pth
if __name__ == "__main__":
    MODEL_PATH = "ckpt/RWKV-x060-World-7B-v2.1-20240507-ctx4096.pth"  # 30.57

    model_params = torch.load(MODEL_PATH, map_location="cpu")

    with torch.no_grad():

        # model = RWKV(args).bfloat16().cuda()
        # model.load_state_dict(model_params)
        args = get_args(MODEL_PATH)
        model = get_mod_rwkv_model(MODEL_PATH)

        prompt = "The Eiffel tower is in the city of"
        input = tokenizer.encode(prompt)
        print(f"\nInput:\n{input}")

        import time

        tick = time.time()
        out = model.forward(torch.tensor(input).reshape(1, -1).cuda())
        print("time %.2f" % (time.time() - tick))
        print(f"\nOutput:\n{out}")

        # let's check the logits for the last token => prediction for the next token

        out = out[0, -1]  # out.shape = [batch_size(B), seq_len(T), n_emb(C)], so out[0,-1] is the logits for the last token

        probs = F.softmax(out.float(), dim=-1)  # compute softmax in float (more accurate)

        print(f"\n{prompt}")

        _, indices = torch.topk(probs, 10)  # print top-10 possibilities
        for i in range(len(indices)):
            token_id = indices[i].item()
            token = tokenizer.decode([token_id])
            token_prob = probs[token_id].item()
            print(token, f"[probability {token_prob:.2%}]")

        with open(f"models/lambada_test.jsonl", "r", encoding="utf-8") as f:
            todo = [json.loads(line) for line in f]
            todo = [[doc["text"].rsplit(" ", 1)[0], " " + doc["text"].rsplit(" ", 1)[1]] for doc in todo]

        print("\nCheck LAMBADA...")
        xsum = 0
        xcnt = 0
        xacc = 0

        total_time = 0
        total_ind = 0

        for index, d in enumerate(todo):
            src = [0] + tokenizer.encode(d[0])
            dst = tokenizer.encode(d[1])

            logits = 0
            correct = True
            if index >= 100:
                tick = time.time()

            out = model.forward(torch.tensor(src + dst).reshape(1, -1).cuda())
            if index >= 100:
                total_time += time.time() - tick
                total_ind += 1

            if index in [300, 500, 1000]:
                print(index, "time %.2f" % (total_ind / total_time))

            if index == 1000:
                break

            for i in range(len(dst)):
                ooo = out[0, len(src) - 1 + i].float()
                probs = F.softmax(ooo, dim=-1)
                logits += math.log(probs[dst[i]])
                if torch.argmax(probs).item() != dst[i]:
                    correct = False
            xcnt += 1
            xsum += logits
            xacc += 1 if correct else 0
            if xcnt % 100 == 0 or xcnt == len(todo):
                print(xcnt, "ppl", round(math.exp(-xsum / xcnt), 2), "acc", round(xacc / xcnt * 100, 2))
