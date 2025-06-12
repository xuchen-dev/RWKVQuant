########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import torch, types, os, gc, math, json
import numpy as np
import torch.nn as nn
from torch.nn import functional as F
from quant.rotation_utils import get_orthogonal_matrix

np.set_printoptions(precision=4, suppress=True, linewidth=200)

"""
This will load RWKV-7 "Goose" x070 and inference in GPT-mode (slower than RNN-mode for autoregressive generation)
"""
USE_CUDA_KERNEL = True  # False => UNOPTIMIZED, VERY SLOW
DTYPE = torch.half
head_size_a = 64  # don't change
HEAD_SIZE = head_size_a


def get_args(model_path):
    args = types.SimpleNamespace()

    # model download: https://huggingface.co/BlinkDL/rwkv-7-pile
    MODEL_PATH = model_path
    # MODEL_PATH = "/data01/home/xuchen/gptvq/RWKV/rwkv-7-pile/RWKV-x070-Pile-168M-20241120-ctx4096.pth"
    # MODEL_PATH = "/data01/home/xuchen/gptvq/RWKV/rwkv-7-pile/RWKV-x070-Pile-421M-20241127-ctx4096.pth"
    # MODEL_PATH = "/data01/home/xuchen/gptvq/RWKV/rwkv-7-pile/RWKV-x070-Pile-1.47B-20241210-ctx4096.pth"

    if "168M" in MODEL_PATH:
        args.n_layer = 12
        args.n_embd = 768
        args.D_DECAY_LORA = 64
        args.D_AAA_LORA = 64
        args.D_MV_LORA = 32
        args.D_GATE_LORA = 128
    elif "421M" in MODEL_PATH:
        args.n_layer = 24
        args.n_embd = 1024
        args.D_DECAY_LORA = 64
        args.D_AAA_LORA = 64
        args.D_MV_LORA = 64
        args.D_GATE_LORA = 128
    elif "1.47B" in MODEL_PATH:
        args.n_layer = 24
        args.n_embd = 2048
        args.D_DECAY_LORA = 96
        args.D_AAA_LORA = 96
        args.D_MV_LORA = 64
        args.D_GATE_LORA = 256

    args.head_size_a = 64
    args.vocab_size = 50304  # "pile" model: 50277 padded to 50304
    return args


from tokenizers import Tokenizer

tokenizer = Tokenizer.from_file("/data01/home/xuchen/RWKV-LM/RWKV-v4neo/20B_tokenizer.json")


class Cus_Mul(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.weight = weight

    def forward(self, x1, x2):
        return x1 * x2


########################################################################################################
# CUDA Kernel
########################################################################################################

if USE_CUDA_KERNEL:

    from torch.utils.cpp_extension import load

    load(
        name="wkv7",
        sources=["/data01/home/xuchen/gptvq/cuda/wkv7_op.cpp", f"/data01/home/xuchen/gptvq/cuda/wkv7.cu"],
        is_python_module=False,
        verbose=True,
        extra_cuda_cflags=["-res-usage", "--use_fast_math", "-O3", "-Xptxas -O3", "--extra-device-vectorization", f"-D_N_={HEAD_SIZE}"],
    )

    class WKV_7(torch.autograd.Function):
        @staticmethod
        def forward(ctx, r, w, k, v, a, b):
            with torch.no_grad():
                B, T, C = r.size()
                H = C // HEAD_SIZE
                N = HEAD_SIZE
                assert HEAD_SIZE == C // H
                assert r.dtype == DTYPE
                assert w.dtype == DTYPE
                assert k.dtype == DTYPE
                assert v.dtype == DTYPE
                assert a.dtype == DTYPE
                assert b.dtype == DTYPE
                assert r.is_contiguous()
                assert w.is_contiguous()
                assert k.is_contiguous()
                assert v.is_contiguous()
                assert a.is_contiguous()
                assert b.is_contiguous()
                y = torch.empty((B, T, C), device=k.device, dtype=DTYPE, memory_format=torch.contiguous_format)
                torch.ops.wkv7.forward(B, T, C, H, r, w, k, v, a, b, y)
                return y

    def RWKV7_OP(r, w, k, v, a, b):
        return WKV_7.apply(r, w, k, v, a, b)

else:

    def RWKV7_OP(r, w, k, v, a, b):
        B, T, C = r.size()
        H = C // HEAD_SIZE
        N = HEAD_SIZE
        r = r.view(B, T, H, N).float()
        k = k.view(B, T, H, N).float()
        v = v.view(B, T, H, N).float()
        a = a.view(B, T, H, N).float()
        b = b.view(B, T, H, N).float()
        w = torch.exp(-torch.exp(w.view(B, T, H, N).float()))
        out = torch.zeros((B, T, H, N), device=r.device, dtype=torch.float)
        state = torch.zeros((B, H, N, N), device=r.device, dtype=torch.float)

        for t in range(T):
            kk = k[:, t, :].view(B, H, 1, N)
            rr = r[:, t, :].view(B, H, N, 1)
            vv = v[:, t, :].view(B, H, N, 1)
            aa = a[:, t, :].view(B, H, N, 1)
            bb = b[:, t, :].view(B, H, 1, N)
            state = state * w[:, t, :, None, :] + state @ aa @ bb + vv @ kk
            out[:, t, :] = (state @ rr).view(B, H, N)

            # another method using einsum
            #
            # kk = k[:, t, :]
            # rr = r[:, t, :]
            # vv = v[:, t, :]
            # aa = a[:, t, :]
            # bb = b[:, t, :]
            # sab = torch.einsum('bhik,bhk,bhj->bhij', state, aa, bb)
            # state = state * w[: , t, :, None, :] + sab + torch.einsum('bhj,bhi->bhij', kk, vv)
            # out[:, t, :] = torch.einsum('bhj,bhij->bhi', rr, state)

        return out.view(B, T, C).to(dtype=DTYPE)


########################################################################################################
# RWKV TimeMix
########################################################################################################


class RWKV_Tmix_x070(nn.Module):
    def __init__(self, args, layer_id):
        super().__init__()
        D_DECAY_LORA = args.D_DECAY_LORA
        D_AAA_LORA = args.D_AAA_LORA
        D_MV_LORA = args.D_MV_LORA
        D_GATE_LORA = args.D_GATE_LORA

        self.args = args
        self.layer_id = layer_id

        self.head_size = args.head_size_a  # 64
        self.n_head = args.dim_att // self.head_size
        assert args.dim_att % self.n_head == 0

        H = self.n_head  # 12
        N = self.head_size  # 64
        C = args.n_embd  # 768
        self.c = args.n_embd

        self.x_r = nn.Parameter(torch.empty(1, 1, C))
        self.x_w = nn.Parameter(torch.empty(1, 1, C))
        self.x_k = nn.Parameter(torch.empty(1, 1, C))
        self.x_v = nn.Parameter(torch.empty(1, 1, C))
        self.x_a = nn.Parameter(torch.empty(1, 1, C))
        self.x_g = nn.Parameter(torch.empty(1, 1, C))

        self.w0 = nn.Parameter(torch.empty(1, 1, C))
        self.w1 = nn.Parameter(torch.empty(C, D_DECAY_LORA))  # ,64
        self.w2 = nn.Parameter(torch.empty(D_DECAY_LORA, C))

        self.a0 = nn.Parameter(torch.empty(1, 1, C))
        self.a1 = nn.Parameter(torch.empty(C, D_AAA_LORA))  #  ,64
        self.a2 = nn.Parameter(torch.empty(D_AAA_LORA, C))

        if layer_id > 0:
            self.v0 = nn.Parameter(torch.empty(1, 1, C))
            self.v1 = nn.Parameter(torch.empty(C, D_MV_LORA))
            self.v2 = nn.Parameter(torch.empty(D_MV_LORA, C))

        self.g1 = nn.Parameter(torch.empty(C, D_GATE_LORA))
        self.g2 = nn.Parameter(torch.empty(D_GATE_LORA, C))

        self.k_k = nn.Parameter(torch.empty(1, 1, C))
        self.k_a = nn.Parameter(torch.empty(1, 1, C))
        self.r_k = nn.Parameter(torch.empty(H, N))

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))  # 向下平移1， 截取掉最地下一行的
        self.receptance = nn.Linear(C, C, bias=False)
        self.key = nn.Linear(C, C, bias=False)
        self.value = nn.Linear(C, C, bias=False)
        self.output = nn.Linear(C, C, bias=False)
        self.ln_x = nn.GroupNorm(H, C, eps=64e-5)  # !!! notice eps value !!!

    def forward(self, x, v_first):
        B, T, C = x.size()  # 1 10 768
        H = self.n_head  # 12
        xx = self.time_shift(x) - x

        xr = x + xx * self.x_r
        xw = x + xx * self.x_w
        xk = x + xx * self.x_k
        xv = x + xx * self.x_v
        xa = x + xx * self.x_a
        xg = x + xx * self.x_g

        r = self.receptance(xr)
        w = -F.softplus(-(self.w0 + torch.tanh(xw @ self.w1) @ self.w2)) - 0.5  # soft-clamp to (-inf, -0.5)
        k = self.key(xk)
        v = self.value(xv)
        if self.layer_id == 0:
            v_first = v  # store the v of the first layer
        else:
            v = v + (v_first - v) * torch.sigmoid(self.v0 + (xv @ self.v1) @ self.v2)  # add value residual
        a = torch.sigmoid(self.a0 + (xa @ self.a1) @ self.a2)  # a is "in-context learning rate"
        g = torch.sigmoid(xg @ self.g1) @ self.g2

        kk = k * self.k_k
        kk = F.normalize(kk.view(B, T, H, -1), dim=-1, p=2.0).view(B, T, C)
        k = k * (1 + (a - 1) * self.k_a)

        x = RWKV7_OP(r, w, k, v, -kk, kk * a)
        x = self.ln_x(x.view(B * T, C)).view(B, T, C)

        x = x + ((r.view(B, T, H, -1) * k.view(B, T, H, -1) * self.r_k).sum(dim=-1, keepdim=True) * v.view(B, T, H, -1)).view(B, T, C)
        x = self.output(x * g)
        return x, v_first


########################################################################################################
# RWKV TimeMix
########################################################################################################


class RWKV_Tmix_x070_Mod(nn.Module):
    def __init__(self, ori_m, args):
        super().__init__()
        D_DECAY_LORA = args.D_DECAY_LORA
        D_AAA_LORA = args.D_AAA_LORA
        D_MV_LORA = args.D_MV_LORA
        D_GATE_LORA = args.D_GATE_LORA

        self.args = ori_m.args
        self.layer_id = ori_m.layer_id

        self.head_size = ori_m.head_size  # 64
        self.n_head = ori_m.n_head

        H = ori_m.n_head  # 12
        N = ori_m.head_size  # 64
        C = ori_m.c  # 768  self.c

        self.x_r = ori_m.x_r
        self.x_w = ori_m.x_w
        self.x_k = ori_m.x_k
        self.x_v = ori_m.x_v
        self.x_a = ori_m.x_a
        self.x_g = ori_m.x_g

        self.x_r_mul = Cus_Mul(self.x_r)
        self.x_w_mul = Cus_Mul(self.x_w)
        self.x_k_mul = Cus_Mul(self.x_k)
        self.x_v_mul = Cus_Mul(self.x_v)
        self.x_a_mul = Cus_Mul(self.x_a)
        self.x_g_mul = Cus_Mul(self.x_g)

        # self.w0 = nn.Parameter(torch.empty(1,1,C))
        # self.w1 = nn.Parameter(torch.empty(C, D_DECAY_LORA)) # ,64
        # self.w2 = nn.Parameter(torch.empty(D_DECAY_LORA, C))

        # dev = self.linear_w1.weight.bias.data.device

        self.linear_w1 = nn.Linear(C, D_DECAY_LORA, bias=False)
        self.linear_w1.weight = ori_m.w1
        self.linear_w1.weight.data = self.linear_w1.weight.data.T
        # self.linear_w1.weight.bias.data = \
        #     torch.zeros_like(self.linear_w1.weight.bias.data, device=dev)

        self.linear_w2 = nn.Linear(D_DECAY_LORA, C)
        self.linear_w2.bias = ori_m.w0
        self.linear_w2.weight = ori_m.w2
        self.linear_w2.weight.data = self.linear_w2.weight.data.T
        self.linear_w2.bias.data = self.linear_w2.bias.data.squeeze(0)

        # self.a0 = nn.Parameter(torch.empty(1,1,C))
        # self.a1 = nn.Parameter(torch.empty(C, D_AAA_LORA)) #  ,64
        # self.a2 = nn.Parameter(torch.empty(D_AAA_LORA, C))

        self.linear_a1 = nn.Linear(C, D_AAA_LORA, bias=False)
        self.linear_a1.weight = ori_m.a1
        self.linear_a1.weight.data = self.linear_a1.weight.data.T
        # self.linear_a1.weight.bias.data = \
        #     torch.zeros_like(self.linear_a1.weight.bias.data, device=dev)

        self.linear_a2 = nn.Linear(D_AAA_LORA, C)
        self.linear_a2.bias = ori_m.a0
        self.linear_a2.weight = ori_m.a2
        self.linear_a2.weight.data = self.linear_a2.weight.data.T
        self.linear_a2.bias.data = self.linear_a2.bias.data.squeeze(0)

        if self.layer_id > 0:
            # self.v0 = nn.Parameter(torch.empty(1,1,C))
            # self.v1 = nn.Parameter(torch.empty(C, D_MV_LORA))
            # self.v2 = nn.Parameter(torch.empty(D_MV_LORA, C))

            self.linear_v1 = nn.Linear(C, D_MV_LORA, bias=False)
            self.linear_v1.weight = ori_m.v1
            self.linear_v1.weight.data = self.linear_v1.weight.data.T
            # self.linear_v1.weight.bias.data = \
            #     torch.zeros_like(self.linear_v1.weight.bias.data, device=dev)

            self.linear_v2 = nn.Linear(D_MV_LORA, C)
            self.linear_v2.bias = ori_m.v0
            self.linear_v2.weight = ori_m.v2
            self.linear_v2.weight.data = self.linear_v2.weight.data.T
            self.linear_v2.bias.data = self.linear_v2.bias.data.squeeze(0)

        # self.g1 = nn.Parameter(torch.empty(C, D_GATE_LORA))
        # self.g2 = nn.Parameter(torch.empty(D_GATE_LORA, C))

        self.linear_g1 = nn.Linear(C, D_GATE_LORA, bias=False)
        self.linear_g1.weight = ori_m.g1
        self.linear_g1.weight.data = self.linear_g1.weight.data.T

        self.linear_g2 = nn.Linear(D_GATE_LORA, C, bias=False)
        self.linear_g2.weight = ori_m.g2
        self.linear_g2.weight.data = self.linear_g2.weight.data.T

        self.k_k = ori_m.k_k
        self.k_a = ori_m.k_a
        self.r_k = ori_m.r_k

        self.k_k_mul = Cus_Mul(self.k_k)
        self.k_a_mul = Cus_Mul(self.k_a)
        self.r_k_mul = Cus_Mul(self.r_k)

        self.time_shift = ori_m.time_shift  # 向下平移1， 截取掉最地下一行的
        self.receptance = ori_m.receptance
        self.key = ori_m.key
        self.value = ori_m.value
        self.output = ori_m.output
        self.ln_x = ori_m.ln_x  # !!! notice eps value !!!

    def get_state_params(self):
        params = [
            self.x_r,
            self.x_w,
            self.x_k,
            self.x_v,
            self.x_a,
            self.x_g,
            self.k_k,
            self.k_a,
            self.r_k,
        ]
        return params

    def _rotate(self):
        if hasattr(self, "linear_v1"):
            Q = get_orthogonal_matrix(self.linear_v1.weight.shape[0], mode="hadamard")
            dtype = self.linear_v1.weight.data.dtype
            W1 = self.linear_v1.weight.data.to(dtype=torch.float64)
            self.linear_v1.weight.data = torch.matmul(Q.T, W1).to(dtype=dtype)

            W2 = self.linear_v2.weight.data.to(dtype=torch.float64)
            self.linear_v2.weight.data = torch.matmul(W2, Q).to(dtype=dtype)

        Q_a = get_orthogonal_matrix(self.linear_a1.weight.shape[0], mode="hadamard")
        dtype = self.linear_a1.weight.data.dtype
        W1_a = self.linear_a1.weight.data.to(dtype=torch.float64)
        self.linear_a1.weight.data = torch.matmul(Q_a.T, W1_a).to(dtype=dtype)

        W2_a = self.linear_a2.weight.data.to(dtype=torch.float64)
        self.linear_a2.weight.data = torch.matmul(W2_a, Q_a).to(dtype=dtype)

    def forward(self, x, v_first):
        B, T, C = x.size()  # 1 10 768
        H = self.n_head  # 12
        xx = self.time_shift(x) - x

        xr = x + self.x_r_mul(xx, self.x_r)
        xw = x + self.x_w_mul(xx, self.x_w)
        xk = x + self.x_k_mul(xx, self.x_k)
        xv = x + self.x_v_mul(xx, self.x_v)
        xa = x + self.x_a_mul(xx, self.x_a)
        xg = x + self.x_g_mul(xx, self.x_g)

        r = self.receptance(xr)

        # w = -F.softplus(-(self.w0 + torch.tanh(xw @ self.w1) @ self.w2)) - 0.5 # soft-clamp to (-inf, -0.5)
        w = -F.softplus(-self.linear_w2(torch.tanh(self.linear_w1(xw)))) - 0.5

        k = self.key(xk)
        v = self.value(xv)
        if self.layer_id == 0:
            v_first = v  # store the v of the first layer
        else:
            # v = v + (v_first - v) * torch.sigmoid(self.v0 + (xv @ self.v1) @ self.v2) # add value residual
            v = v + (v_first - v) * torch.sigmoid(self.linear_v2(self.linear_v1(xv)))

        # a = torch.sigmoid(self.a0 + (xa @ self.a1) @ self.a2) # a is "in-context learning rate"
        a = torch.sigmoid(self.linear_a2(self.linear_a1(xa)))  # a is "in-context learning rate"
        # g = torch.sigmoid(xg @ self.g1) @ self.g2
        g = self.linear_g2(torch.sigmoid(self.linear_g1(xg)))

        kk = self.k_k_mul(k, self.k_k)
        kk = F.normalize(kk.view(B, T, H, -1), dim=-1, p=2.0).view(B, T, C)
        # k = k * (1 + (a-1) * self.k_a)
        k = k * (1 + self.k_a_mul((a - 1), self.k_a))

        x = RWKV7_OP(r, w, k, v, -kk, kk * a)
        x = self.ln_x(x.view(B * T, C)).view(B, T, C)

        # x = x + ((r.view(B,T,H,-1)*k.view(B,T,H,-1)*self.r_k).sum(dim=-1, keepdim=True) * v.view(B,T,H,-1)).view(B,T,C)
        x = x + (self.r_k_mul(r.view(B, T, H, -1) * k.view(B, T, H, -1), self.r_k).sum(dim=-1, keepdim=True) * v.view(B, T, H, -1)).view(B, T, C)
        x = self.output(x * g)
        return x, v_first


########################################################################################################
# RWKV ChannelMix
########################################################################################################


class RWKV_CMix_x070(nn.Module):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        with torch.no_grad():
            self.x_k = nn.Parameter(torch.empty(1, 1, args.n_embd))

        self.key = nn.Linear(args.n_embd, args.dim_ffn, bias=False)
        self.value = nn.Linear(args.dim_ffn, args.n_embd, bias=False)

    def forward(self, x):
        xx = self.time_shift(x) - x

        k = x + xx * self.x_k
        k = torch.relu(self.key(k)) ** 2
        return self.value(k)


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

        self.att = RWKV_Tmix_x070(args, layer_id)
        self.ffn = RWKV_CMix_x070(args, layer_id)

    def forward(self, x, v_first):

        if self.layer_id == 0:
            x = self.ln0(x)  # [1,10,768]

        xx, v_first = self.att(self.ln1(x), v_first)
        x = x + xx
        x = x + self.ffn(self.ln2(x))

        return x, v_first


########################################################################################################
# RWKV Model
########################################################################################################


class RWKV(nn.Module):
    def __init__(self, args):
        super().__init__()
        args.dim_att = args.n_embd
        args.dim_ffn = args.n_embd * 4
        self.emb = nn.Embedding(args.vocab_size, args.n_embd)

        self.blocks = nn.ModuleList([Block(args, i) for i in range(args.n_layer)])

        self.ln_out = nn.LayerNorm(args.n_embd)
        self.head = nn.Linear(args.n_embd, args.vocab_size, bias=False)

    def forward(self, idx):

        x = self.emb(idx)

        v_first = torch.empty_like(x)
        for block in self.blocks:
            x, v_first = block(x, v_first)  # [1,10,768]

        x = self.ln_out(x)
        x = self.head(x)

        return x


########################################################################################################
# RWKV utils
########################################################################################################


def get_mod_rwkv_model(model_path=None, rotation=False):
    args = get_args(model_path)
    model_params = torch.load(model_path, map_location="cpu")
    model = RWKV(args).to(dtype=DTYPE).cuda()
    model.load_state_dict(model_params, strict=False)

    for index, block in enumerate(model.blocks):
        model.blocks[index].att = RWKV_Tmix_x070_Mod(block.att, args)
        if rotation:
            model.blocks[index].att._rotate()

    return model


def get_rwkv_dataset(n_samples=-1):
    with open(f"/data01/home/xuchen/RWKV-LM/RWKV-v7/misc/lambada_test.jsonl", "r", encoding="utf-8") as f:
        todo = [json.loads(line) for line in f]
        todo = [[doc["text"].rsplit(" ", 1)[0], " " + doc["text"].rsplit(" ", 1)[1]] for doc in todo]
    if n_samples > 0:
        return todo[:n_samples]
    return todo


def get_rwkv_token():
    tokenizer = Tokenizer.from_file("/data01/home/xuchen/RWKV-LM/RWKV-v4neo/20B_tokenizer.json")
    return tokenizer


def eval_lambada(dataset, model, tokenizer, logger):
    print("\nCheck LAMBADA...")
    xsum = 0
    xcnt = 0
    xacc = 0
    for d in dataset:
        src = [0] + tokenizer.encode(d[0]).ids
        dst = tokenizer.encode(d[1]).ids

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


def eval_lambada(dataset, model, tokenizer, logger):
    print("\nCheck LAMBADA...")
    xsum = 0
    xcnt = 0
    xacc = 0
    for d in dataset:
        src = [0] + tokenizer.encode(d[0]).ids
        dst = tokenizer.encode(d[1]).ids

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


def eval_ppl_(model, test_loader, seqlen=-1, limit=-1):
    nlls = []
    nsamples = test_loader.numel() // seqlen

    for i in tqdm(range(nsamples)):
        batch = test_loader[:, (i * seqlen) : ((i + 1) * seqlen)].to(model.device)
        net_name = model.name.lower() if hasattr(model, "name") else type(model).__name__.lower()
        if "opt" in net_name:
            outputs = model.model.model.decoder(batch)
            hidden_states = outputs[0]
            logits = model.model.lm_head(hidden_states)
        elif "llama" in net_name or "mixtral" in net_name:
            # import pdb;pdb.set_trace()
            outputs = model(batch)
            logits = outputs["logits"]
            outputs = None
        elif "falcon" in net_name:
            outputs = model.model.transformer(batch)
            hidden_states = outputs[0]
            logits = model.model.lm_head(hidden_states)
        elif "glm" in net_name:
            outputs = model(batch)
            logits = outputs["logits"]
            outputs = None
        shift_logits = logits[:, :-1, :]
        shift_labels = test_loader[:, (i * seqlen) : ((i + 1) * seqlen)][:, 1:].to(logits.device)
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )
        neg_log_likelihood = loss.float() * seqlen
        nlls.append(neg_log_likelihood)
        if i == limit:
            break
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * seqlen))
    return ppl.item()


def eval_ppl(model, tokenizer, seqlen=2048, limit=-1):
    model.eval()
    wiki_testdata = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    wiki_testloader = tokenizer("\n\n".join(wiki_testdata["text"]), return_tensors="pt")
    wiki_ppl = eval_ppl_(model, wiki_testloader.input_ids, seqlen, limit)
    print(f"wiki ppl : {wiki_ppl}")

    c4_testdata = load_dataset("allenai/c4", data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"}, split="validation")
    random.seed(0)
    valenc = []
    for _ in range(256):
        while True:
            i = random.randint(0, len(c4_testdata) - 1)
            tmp = tokenizer(c4_testdata[i]["text"], return_tensors="pt")
            if tmp.input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, tmp.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        valenc.append(tmp.input_ids[:, i:j])
    c4_testloader = torch.hstack(valenc)
    c4_ppl = eval_ppl_(model, c4_testloader, seqlen, limit)
    print(f"c4 ppl : {c4_ppl}")


########################################################################################################
# RWKV Inference
########################################################################################################

if __name__ == "__main__":
    MODEL_PATH = "/data01/home/xuchen/gptvq/RWKV/rwkv-7-pile/RWKV-x070-Pile-168M-20241120-ctx4096.pth"

    args = get_args(MODEL_PATH)
    model_params = torch.load(MODEL_PATH, map_location="cpu")

    with torch.no_grad():

        model = RWKV(args).to(dtype=DTYPE).cuda()
        model.load_state_dict(model_params)

        ########################################################################################################

        prompt = "The Eiffel tower is in the city of"
        input = tokenizer.encode(prompt).ids
        print(f"\nInput:\n{input}")

        out = model.forward(torch.tensor(input).reshape(1, -1).cuda())
        print(f"\nOutput:\n{out}")

        ################################################  替换模型
        for index, block in enumerate(model.blocks):
            model.blocks[index].att = RWKV_Tmix_x070_Mod(block.att, args)

        out_mod = model.forward(torch.tensor(input).reshape(1, -1).cuda())
        print((out - out_mod).abs().mean())
        ################################################

        # logits of the last token => prediction for the next token
        out = out[0, -1]

        probs = F.softmax(out.float(), dim=-1)  # compute softmax in float (more accurate)

        print(f"\n{prompt}")

        _, indices = torch.topk(probs, 10)  # print top-10 possibilities
        for i in range(len(indices)):
            token_id = indices[i].item()
            token = tokenizer.decode([token_id])
            token_prob = probs[token_id].item()
            print(token, f"[probability {token_prob:.2%}]")

        ########################################################################################################

        with open(f"/data01/home/xuchen/RWKV-LM/RWKV-v7/misc/lambada_test.jsonl", "r", encoding="utf-8") as f:
            todo = [json.loads(line) for line in f]
            todo = [[doc["text"].rsplit(" ", 1)[0], " " + doc["text"].rsplit(" ", 1)[1]] for doc in todo]

        print("\nCheck LAMBADA...")
        xsum = 0
        xcnt = 0
        xacc = 0
        for d in todo:
            src = [0] + tokenizer.encode(d[0]).ids
            dst = tokenizer.encode(d[1]).ids

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
            if xcnt % 100 == 0 or xcnt == len(todo):
                print(xcnt, "ppl", round(math.exp(-xsum / xcnt), 2), "acc", round(xacc / xcnt * 100, 2))
