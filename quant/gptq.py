import torch
from torch import nn
import numpy as np
import math
import time
import transformers
from quant.quant import *
from quant.vq_quant import vq_quantize, quantize_centroids, vq_quantize_mul
from quant.utils import draw_img
from copy import deepcopy
from quant.rotation_utils import get_orthogonal_matrix

DEBUG = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


global flops_number
flops_number = 0


def get_extr_flops():
    global flops_number
    return flops_number


def add_extr_flops(flops):
    global flops_number
    flops_number += flops


def quad_loss(w_q, G, v, offset):
    """
    A generic function for computing the quadratic loss:
    L = 1/2 (G w_q, w_q) + (v, w_q) + offset

    Parameters
    ----------
    w_q : (c_out, m) or (m, 1)
        Quantized weights to be optimized.
    G : (m, m)
        Matrix part.
    v : shape(w_q)
        Linear part.
    offset : ()
        Scalar part.
    """
    # Quadratic loss: 1/2 wGw^T
    loss = 0.5 * (w_q.mm(G) * w_q).sum()
    # Add linear term and offset
    loss += (v * w_q).sum()
    loss += offset
    return loss


def quad_loss_2(W, Q, G):
    Werr = W - Q
    return (Werr.mm(G) * Werr).sum()


class GPTQ:

    def __init__(self, layer):
        self.layer = layer
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.rows = W.shape[0]  # H
        self.columns = W.shape[1]  # W
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)  # W，W
        self.nsamples = 0

    def add_batch(self, inp, out):
        self.inp1 = inp
        self.out1 = out
        if DEBUG:
            self.inp1 = inp
            self.out1 = out
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear) or isinstance(self.layer, transformers.Conv1D):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()
        if isinstance(self.layer, nn.Conv2d):
            unfold = nn.Unfold(
                self.layer.kernel_size,
                dilation=self.layer.dilation,
                padding=self.layer.padding,
                stride=self.layer.stride,
            )
            inp = unfold(inp)
            inp = inp.permute([1, 0, 2])
            inp = inp.flatten(1)

        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        self.H += inp.matmul(inp.t())

    def lut_m_step(self, Q_orig, groupsize, quantizer, scale=None, svd_rank=None, logger=None, mquantizers=None):
        with torch.enable_grad():
            W = self.layer.weight.data.clone().float()
            G = self.G
            del self.G
            if scale is not None:
                scale.detach()

            offset = (W.mm(G) * W).sum()

            all_centroids = quantizer.all_centroids
            all_assignments = self.assignments
            vq_dim = quantizer.vq_dim

            if svd_rank is not None:
                assert vq_dim == 1, "In this implementation, SVD only works on 1D VQ"
                r = int(all_centroids[0].shape[1] * svd_rank)
                print(f"Effective SVD rank: {r}")
                Groups = all_centroids[0].shape[0]

                C = torch.concat(all_centroids, dim=0).squeeze()  # G x K
                C, new_idx = torch.sort(C, dim=1)
                new_idx = torch.argsort(new_idx, dim=1).split(Groups)

                U, S, V = torch.linalg.svd(C, full_matrices=False)
                all_centroids, V = (U * S[None])[:, :r].split(Groups), V[:r]

                new_assignments = []
                for idx, a in zip(new_idx, all_assignments):
                    new_assignments.append([])
                    for a_ in a:
                        remapped_a = torch.gather(idx, dim=1, index=a_)
                        new_assignments[-1].append(remapped_a)
                all_assignments = new_assignments

            def make_quantized_weight(centroids, assignments, scale=None, mquant=None):
                all_values = []
                for id, (c, a) in enumerate(zip(centroids, assignments)):
                    if len(a) == 0:
                        index, mquantizer = mquant[id]
                        values = mquantizer.quantize(self.layer.weight.data[:, index[0] : index[1]])
                        all_values.append(values.view(W.shape[0], -1))
                        continue

                    if svd_rank is not None:
                        c = (c @ V).unsqueeze(-1)
                    for a_ in a:
                        values = torch.gather(c, dim=1, index=a_.unsqueeze(-1).expand(-1, -1, vq_dim))
                        all_values.append(values.view(W.shape[0], -1))

                Q = torch.concat(all_values, dim=1)
                if scale is not None:
                    Q = torch.mul(Q, scale)
                return Q

            with torch.no_grad():
                Q = make_quantized_weight(all_centroids, all_assignments, scale, mquant=mquantizers)

                orig_loss = quad_loss_2(W, Q, G)
                snr_before = 10 * np.log10(offset.item() / orig_loss.item())

            must_restart = True
            lr = 1e-3

            while must_restart:
                orig_centroids = [c.data.clone() for c in all_centroids]

                [c.requires_grad_() for c in all_centroids]
                param_list = list(all_centroids) + ([] if svd_rank is None else [V])

                o = torch.optim.Adam(param_list, lr=lr)
                for _ in range(25):
                    must_restart = False
                    o.zero_grad()
                    Q = make_quantized_weight(all_centroids, all_assignments, scale, mquant=mquantizers)
                    loss = quad_loss_2(W, Q, G)
                    if loss > orig_loss or torch.isnan(loss):
                        lr *= 1e-1
                        print(f"Inner loop: Restarting M-step with lr={lr:.2e}")
                        must_restart = True
                        all_centroids = orig_centroids
                        break
                    loss.backward()
                    o.step()

                if not must_restart:
                    if quantizer.codebook_bitwidth is not None:
                        new_all_centroids = [
                            quantize_centroids(
                                c.requires_grad_(False),
                                quantizer.codebook_bitwidth,
                                per_codebook=quantizer.quantize_per_codebook,
                            )
                            for c in all_centroids
                        ]
                    else:
                        new_all_centroids = all_centroids
                    Q = make_quantized_weight(new_all_centroids, all_assignments, scale, mquant=mquantizers)
                    loss = quad_loss_2(W, Q, G)
                    if torch.isnan(loss):
                        lr *= 1e-1
                        print(f"Outer loop: Restarting M-step with lr={lr:.2e}")
                        must_restart = True
                        all_centroids = orig_centroids
                        continue

                    del orig_centroids
                    # print(
                    #     f"time M-step SGD {(time.time() - self.tick):.2f}; final loss: {loss.item():.4f}"
                    # )
                    logger.info(f"time M-step SGD {(time.time() - self.tick):.2f}; final loss: {loss.item():.4f}")
                    orig_loss = quad_loss_2(W, Q, G)
                    snr_after = 10 * np.log10(offset.item() / orig_loss.item())

                    print(f"improvement: {snr_before:.2f} -> {snr_after:.2f}")

            # Q_step1 = deepcopy(Q)

            if False:
                must_restart = True
                lr = 1e-5
                while must_restart and mquantizers:
                    orgi_mquant = deepcopy(mquantizers)
                    param_list_scale = []
                    for mquant in mquantizers:
                        if len(mquant) > 0:
                            quant = mquant[1]
                            quant.scale.requires_grad_()
                            param_list_scale.append(quant.scale)

                    param_list = param_list_scale

                    if len(param_list) == 0:
                        return Q

                    o = torch.optim.Adam(param_list, lr=lr)

                    for _ in range(100):
                        must_restart = False
                        o.zero_grad()
                        Q = make_quantized_weight(all_centroids, all_assignments, scale, mquant=mquantizers)
                        loss = quad_loss_2(W, Q, G)
                        if loss > orig_loss + 10 or torch.isnan(loss):
                            lr *= 1e-1
                            print(f"Inner loop: Restarting M-step with lr={lr:.2e}")
                            must_restart = True
                            mquantizers = orgi_mquant
                            break
                        loss.backward()
                        o.step()

                    if loss > orig_loss:
                        return Q_step1

                    if not must_restart:
                        if quantizer.codebook_bitwidth is not None:
                            new_all_centroids = [
                                quantize_centroids(
                                    c.requires_grad_(False),
                                    quantizer.codebook_bitwidth,
                                    per_codebook=quantizer.quantize_per_codebook,
                                )
                                for c in all_centroids
                            ]
                        else:
                            new_all_centroids = all_centroids
                        Q = make_quantized_weight(new_all_centroids, all_assignments, scale, mquant=mquantizers)
                        loss = quad_loss_2(W, Q, G)
                        if torch.isnan(loss):
                            lr *= 1e-1
                            print(f"Outer loop: Restarting M-step with lr={lr:.2e}")
                            must_restart = True
                            mquantizers = orgi_mquant
                            continue

                        del orgi_mquant
                        logger.info(f"time M-step SGD {(time.time() - self.tick):.2f}; final loss: {loss.item():.4f}")
                        orig_loss = quad_loss_2(W, Q, G)
                        snr_after = 10 * np.log10(offset.item() / orig_loss.item())

                        print(f"improvement: {snr_before:.2f} -> {snr_after:.2f}")

        return Q

    def fasterquant(
        self,
        blocksize=128,
        percdamp=0.01,
        groupsize=-1,
        actorder=False,
        static_groups=False,
        include_m_step=False,
        use_vq=False,
        svd_rank=None,
        hessian_weighted_lookups=False,
        only_init_kmeans=False,
        use_incoherent=False,
        incoherent_param=10,
        logger=None,
        args=None,
        coherent_type="incoherent",
        name="layer",
        all_metric_list=None,
        vq_cnt=[0],
        k_ablition=False,
    ):
        # use_vq = True

        W = self.layer.weight.data.clone()  # o i
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float()

        if use_incoherent:  # use_incoherent:
            # molan = get_moran_stat(W.cpu().numpy())
            # print("Moran I:", molan)

            if coherent_type and coherent_type not in "mse":
                # assert incoherent_param > 0
                # U, s, Vh = torch.linalg.svd(H)
                if coherent_type == "var":
                    is_incoherent_number = W.var() * 100
                elif coherent_type == "cv":
                    is_incoherent_number = W.var() / W.abs().mean() * 50
                elif coherent_type == "range":
                    is_incoherent_number = (W.max() - W.min()) * 10
                elif coherent_type == "mad":
                    is_incoherent_number = (W - W.mean()).abs().mean() * 100
                elif coherent_type == "h":
                    W_norm = (W - W.min()) / (W - W.min()).sum()
                    # W_norm = W.abs() / W.abs().sum()
                    is_incoherent_number = -(W_norm * torch.log(W_norm + 1e-12)).sum()
                elif coherent_type == "hce":
                    W_norm = (W - W.min()) / (W - W.min()).sum()
                    # W_norm = W.abs() / W.abs().sum()
                    n = W.numel()
                    M = W_norm - W_norm.mean()
                    is_incoherent_number = -(W_norm * torch.log(W_norm + 1e-12)).sum() - (M**2).sum() * n - (M**3).sum() * (n**2) - (M**4).sum() * (n**3)
                elif coherent_type == "hce_v2":
                    W_sort = W.contiguous().view(-1).sort()[0]
                    G = (W_sort[1:] - W_sort[:-1]).unique()
                    G_norm = G / G.sum()
                    is_incoherent_number = -(G_norm * torch.log(G_norm + 1e-12)).sum()
                elif coherent_type == "incoherent":
                    pass
                else:
                    raise NotImplementedError

                if G_norm[0] == 0:
                    G_norm = G_norm[1:]
                n = G_norm.numel()

                hce = -np.log(1 / n) + ((G_norm) * torch.log(G_norm)).sum()

                a1 = (n**2) / 2
                a2 = (n**3) / 6
                a3 = (n**4) / 12
                if k_ablition:
                    if k_ablition < 3:
                        a3 = 0
                    if k_ablition < 2:
                        a2 = 0
                    if k_ablition < 1:
                        a1 = 0
                    if k_ablition == 4:
                        a4 = (n**5) / 20
                    if k_ablition == 5:
                        a5 = (n**6) / 30

                fine_number = a1 * ((G_norm - 1 / n) ** 2).mean().abs() + a2 * ((G_norm - 1 / n) ** 3).mean().abs() + a3 * ((G_norm - 1 / n) ** 4).mean().abs()

                if k_ablition:
                    if k_ablition == 4:
                        fine_number += a4 * ((G_norm - 1 / n) ** 5).mean().abs()
                    if k_ablition == 5:
                        fine_number += a5 * ((G_norm - 1 / n) ** 6).mean().abs()

                if all_metric_list is not None:
                    all_metric_list[0].append(hce)
                    all_metric_list[1].append(fine_number)

                # return
                # print(name)
                # print(hce)
                # print(fine_number)

                # draw_img(W.detach().cpu().numpy(), "/data01/home/xuchen/gptvq/experiment/a_imgs/latest", name+f"{is_incoherent_number}_{fine_number}"+".png")
                # torch.save(W.detach().cpu().numpy(), "/data01/home/xuchen/gptvq/experiment/a_imgs/latest/layer_"+name+".npy")

                is_incoherent_number = W.abs().max() * (W.abs() ** 2).sum().mean().rsqrt() * np.sqrt(W.shape[0] * W.shape[1])
                # is_coherent = hce<incoherent_param[0] and fine_number<incoherent_param[1] # is_incoherent_number > incoherent_param

                is_incoherent = is_incoherent_number > incoherent_param
                # if is_coherent:
                if not is_incoherent:
                    # draw_img(W.detach().cpu().numpy(), "experiment/a_imgs/incoherent", name+".png")
                    if use_vq:
                        use_vq = False
                        include_m_step = False
                        print("weight is incoherent use m_quant instead of vq_quant")
                        logger.info(f"weight is incoherent use m_quant instead of vq_quant with u={incoherent_param} and incoherent_number {is_incoherent_number}")
                    else:
                        use_vq = True
                        include_m_step = True
                        print("weight is coherent use vq_quant instead of mquant")
                        logger.info(f"weight is coherent use vq_quant instead of mquant")
                    self.quantizer = self.quantizer_bak
                else:
                    vq_cnt[0] += 1
                    # print(f"weight is coherent with u={incoherent_param} and incoherent_number {is_incoherent_number}")
                    logger.info(f"weight is coherent with u={incoherent_param} and incoherent_number {is_incoherent_number}")
                    # draw_img(W.detach().cpu().numpy(), "experiment/a_imgs/coherent", name+".png")
                    # if is_incoherent_number < 5: # 
                    #     self.quantizer.configure(args.wbits-1, perchannel=True, sym=args.sym, mse=False)
                    #     logger.info(f"weight is very coherent {is_incoherent_number}")
            else:
                pass

        # return
        self.tick = time.time()

        if not self.quantizer.ready() and not use_vq:
            self.quantizer.find_params(W, weight=True)

        H = self.H
        self.G = self.H.clone()
        del self.H

        dead = torch.diag(H) == 0
        if dead.sum() != 0:
            print("has dead hession matrix")

        H[dead, dead] = 1
        W[:, dead] = 0

        if static_groups:
            raise NotImplementedError("Static groups are not supported in this repo")

        if actorder:
            raise NotImplementedError("Activation (re)-ordering is not supported in this repo")

        vq_dim = self.assignments = None
        S = vq_scaling_blocksize = vq_scaling_n_bits = None

        if use_vq:
            vq_dim = self.quantizer.vq_dim
            groupsize = self.quantizer.get_groupsize(W, groupsize)
            self.assignments = []
            assert blocksize % vq_dim == 0

            vq_scaling_blocksize = self.quantizer.vq_scaling_blocksize
            vq_scaling_n_bits = self.quantizer.vq_scaling_n_bits
            if vq_scaling_blocksize > 0:
                assert vq_scaling_blocksize % vq_dim == 0
                S = torch.ones_like(W)

            # print(W.shape)
            print(f"VQ scaling BS {vq_scaling_blocksize} @ {vq_scaling_n_bits}b " f"({self.quantizer.vq_scaling_domain} domain)")
            print(f"Using Hessian-aware K-means {hessian_weighted_lookups}")

        Losses = torch.zeros_like(W)
        Q = torch.zeros_like(W)

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H

        mq_count = 0
        vq_count = 0
        mquantizers = []

        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            if use_vq and vq_scaling_blocksize > 0:
                W1_scaled, S1 = self.quantizer.blockwise_normalize_data(
                    W1,
                    vq_scaling_blocksize,
                    self.quantizer.vq_scaling_norm,
                    vq_scaling_n_bits,
                    self.quantizer.vq_scaling_domain,
                )
                S[:, i1:i2] = S1
            else:
                W1_scaled = W1
                S1 = torch.ones_like(W1)

            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            for i in range(count):
                if groupsize != -1:
                    if (i1 + i) % groupsize == 0:

                        # if use_incoherent and args.reset_vq: # old is not open
                        #     use_vq = True # xc

                        extra_args = {}
                        if use_vq:
                            if vq_dim > 1 and hessian_weighted_lookups:
                                H_inv_diag = torch.diag(Hinv)[i1 + i : i1 + i + groupsize]
                                extra_args["H_inv_diag"] = H_inv_diag

                        W_group = W[:, (i1 + i) : (i1 + i + groupsize)]

                        W_group_scaled = W_group

                        if use_vq:
                            self.assignments.append([])
                            if vq_scaling_blocksize > 0:
                                assert vq_scaling_blocksize % vq_dim == 0
                                W_group_scaled, S_group = self.quantizer.blockwise_normalize_data(
                                    W_group,
                                    vq_scaling_blocksize,
                                    self.quantizer.vq_scaling_norm,
                                    self.quantizer.vq_scaling_n_bits,
                                    self.quantizer.vq_scaling_domain,
                                )

                        dist = self.quantizer.find_params(W_group_scaled, weight=True, **extra_args)

                        if False:
                            if coherent_type == "incoherent":
                                assert incoherent_param > 0
                                # U, s, Vh = torch.linalg.svd(H)
                                is_incoherent_number = W_group_scaled.abs().max() * (W_group_scaled.abs() ** 2).sum().mean().rsqrt() * np.sqrt(W_group_scaled.shape[0] * W_group_scaled.shape[1])

                                is_incoherent_number = is_incoherent_number + dist * 1e3  # / 2
                                # print(is_incoherent_number)

                                is_incoherent = is_incoherent_number < incoherent_param

                                if not is_incoherent:
                                    # draw_img(W_group_scaled.detach().cpu().numpy(), "experiment/a_imgs/incoherent_block", name+f"_{i1 + i}.png")
                                    mq_count += 1
                                    if use_vq:
                                        use_vq = False
                                        include_m_step = False
                                        # print("weight is incoherent use m_quant instead of vq_quant")
                                        logger.info(f"weight is incoherent use m_quant instead of vq_quant")
                                    else:
                                        use_vq = True
                                        include_m_step = True
                                        # print("weight is coherent use vq_quant instead of mquant")
                                        logger.info(f"weight is coherent use vq_quant instead of mquant")

                                    id = i1 + i
                                    mquantizer = Quantizer()
                                    mquantizer.configure(args.wbits, perchannel=True, sym=args.sym, mse=args.mq_mse)
                                    mquantizer.find_params(W_group_scaled, weight=True, **extra_args)
                                    # mquantizer.optimize_s()

                                    mquantizers.append(((id, id + groupsize), mquantizer))
                                    self.quantizer_bak = mquantizer

                                else:
                                    mquantizers.append([])
                                    # print(f"weight is coherent with u={incoherent_param} and incoherent_number {is_incoherent_number}")
                                    logger.info(f"weight is coherent with u={incoherent_param} and incoherent_number {is_incoherent_number}")
                                    # draw_img(W_group_scaled.detach().cpu().numpy(), "experiment/a_imgs/coherent_block", name+f"_{i1 + i}.png")
                                    vq_count += 1
                                    # if is_incoherent_number < 5: # 降低一bit
                                    #     self.quantizer.configure(args.wbits-1, perchannel=True, sym=args.sym, mse=False) # 设置中心个数
                                    #     logger.info(f"weight is very coherent {is_incoherent_number}")
                            else:
                                raise NotImplementedError

                        # molan = get_moran_stat(W_group_scaled.cpu().numpy())
                        # is_incoherent_number = W_group_scaled.abs().max() * (W_group_scaled.abs() ** 2).sum().mean().rsqrt() * np.sqrt(W.shape[0]*W.shape[1])

                        # print("is_incoherent_number:", is_incoherent_number)

                        # if molan>0.1:
                        #     print("Moran I is big:", molan, is_incoherent_number)
                        #     draw_img(W_group_scaled.detach().cpu().numpy(), "/data01/home/xuchen/gptvq/experiment/a_imgs/molan/", name+".png")

                        # draw_img(W_group_scaled.detach().cpu().numpy(), "experiment/a_imgs/coherent", name+".png")

                if not use_vq:
                    if use_incoherent:
                        w = W1[:, i]
                        d = Hinv1[i, i]

                        q = quantize(
                            w.unsqueeze(1),
                            self.quantizer_bak.scale,
                            self.quantizer_bak.zero,
                            self.quantizer_bak.maxq,
                        ).flatten()

                        Q1[:, i] = q
                        Losses1[:, i] = (w - q) ** 2 / d**2

                        err1 = (w - q) / d
                        # (R x 1).matmul(1 x C') --> R x C' (C': remaining (unquantized) columns)
                        W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                        Err1[:, i] = err1
                    else:
                        w = W1[:, i]
                        d = Hinv1[i, i]

                        q = quantize(
                            w.unsqueeze(1),
                            self.quantizer.scale,
                            self.quantizer.zero,
                            self.quantizer.maxq,
                        ).flatten()

                        Q1[:, i] = q
                        Losses1[:, i] = (w - q) ** 2 / d**2

                        err1 = (w - q) / d
                        # (R x 1).matmul(1 x C') --> R x C' (C': remaining (unquantized) columns)
                        W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                        Err1[:, i] = err1

                elif i % vq_dim == 0:
                    w = W1[:, i : i + vq_dim]  # R x D  4096*1
                    d = torch.diag(Hinv1)[i : i + vq_dim].unsqueeze(0)  # 1 x D
                    w_scaled = W1_scaled[:, i : i + vq_dim]  # R x D
                    s = S1[:, i : i + vq_dim]

                    H_inv_diag = None
                    if vq_dim > 1 and hessian_weighted_lookups:
                        H_inv_diag = 1.0 / d.to(w.device)

                    q, assmt = vq_quantize(w_scaled, self.quantizer, H_inv_diag=H_inv_diag)  # R x 1 x D, R x 1
                    q = torch.mul(q, s)  # de-scaling

                    self.assignments[-1].append(assmt)

                    Q1[:, i : i + vq_dim] = q
                    Losses1[:, i : i + vq_dim] = (w - q) ** 2 / d**2  # R x D / 1 x D

                    err1 = (w - q) / d  # R x D
                    # batch matmul solution: (D x R x 1).matmul(D x 1 x C').sum(0) --> R x C'
                    if not only_init_kmeans:
                        update = torch.bmm(
                            err1.transpose(0, 1).unsqueeze(-1),
                            Hinv1[i : i + vq_dim, i + vq_dim :].unsqueeze(1),
                        ).sum(0)
                        W1[:, i + vq_dim :] -= update
                        Err1[:, i : i + vq_dim] = err1

            Q[:, i1:i2] = Q1
            Losses[:, i1:i2] = Losses1 / 2

            if not only_init_kmeans:
                W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

            if DEBUG:
                self.layer.weight.data[:, :i2] = Q[:, :i2]
                self.layer.weight.data[:, i2:] = W[:, i2:]
                print(torch.sum((self.layer(self.inp1) - self.out1) ** 2))
                print(torch.sum(Losses))

        torch.cuda.synchronize()
        print("time %.2f" % (time.time() - self.tick))
        # print("error", torch.sum(Losses).item())
        logger.info(f"error:{torch.sum(Losses).item()}")

        # if mq_count:
        #     logger.info(f"mq percent :{ mq_count / (mq_count + vq_count)}")

        # if actorder:
        #     Q = Q[:, invperm]

        if isinstance(self.layer, transformers.Conv1D):
            Q = Q.t()

        if include_m_step:  # and not use_incoherent  [768, 64]
            Q = self.lut_m_step(Q, groupsize, self.quantizer, scale=S, svd_rank=svd_rank, logger=logger, mquantizers=mquantizers)

        if False:  # coherent_type: #  == "mse"
            ori_w = self.layer.weight.data.clone()
            vq_w = Q.clone()
            self.layer.weight.data = Q.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)
            vq_loss = torch.sum((self.layer(self.inp1) - self.out1) ** 2)

            Q = torch.zeros_like(W)
            for i1 in range(0, self.columns, blocksize):
                i2 = min(i1 + blocksize, self.columns)
                count = i2 - i1
                W1 = ori_w[:, i1:i2].clone()

                Q1 = torch.zeros_like(W1)
                Err1 = torch.zeros_like(W1)
                Losses1 = torch.zeros_like(W1)

                for i in range(count):
                    if groupsize != -1:
                        if (i1 + i) % groupsize == 0:
                            W_group = W[:, (i1 + i) : (i1 + i + groupsize)]
                            dist = self.quantizer_bak.find_params(W_group, weight=True, **extra_args)

                    w = W1[:, i]
                    d = Hinv1[i, i]

                    q = quantize(
                        w.unsqueeze(1),
                        self.quantizer_bak.scale,
                        self.quantizer_bak.zero,
                        self.quantizer_bak.maxq,
                    ).flatten()

                    Q1[:, i] = q
                    Losses1[:, i] = (w - q) ** 2 / d**2

                    err1 = (w - q) / d
                    # (R x 1).matmul(1 x C') --> R x C' (C': remaining (unquantized) columns)
                    W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                    Err1[:, i] = err1

                Q[:, i1:i2] = Q1

            self.layer.weight.data = Q.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)
            uq_loss = torch.sum((self.layer(self.inp1) - self.out1) ** 2)

            if uq_loss > vq_loss:
                self.layer.weight.data = vq_w.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)
                logger.info(f"uq_loss:{uq_loss}, vq_loss:{vq_loss}")
            else:
                logger.info(f"uq_loss:{uq_loss}, vq_loss:{vq_loss}")

        self.layer.weight.data = Q.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)
        if DEBUG:
            print(torch.sum((self.layer(self.inp1) - self.out1) ** 2))

    def free(self):
        if DEBUG:
            self.inp1 = None
            self.out1 = None
        self.H = None
        self.Losses = None
        self.Trace = None
        torch.cuda.empty_cache()


class Mask_Q:
    def __init__(self, layer):
        self.layer = layer
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        # print(W.shape)
        if len(W.shape) == 3:
            self.rows = W.shape[1]  # H
            self.columns = W.shape[2]  # W
        else:
            self.rows = W.shape[0]  # H
            self.columns = W.shape[1]  # W
        self.nsamples = 0
        self.mask = None
        self.input_list = []

    def add_batch(self, inp, out):
        if DEBUG:
            self.inp1 = inp
            self.out1 = out

        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        self.input_list.append(inp)

        if self.mask is None:
            self.mask = torch.mean(inp, dim=[0, 1], keepdim=True).abs()
        else:
            self.mask *= self.nsamples / (self.nsamples + tmp)
            self.nsamples += tmp
            self.mask += torch.mean(inp, dim=[0, 1], keepdim=True).abs() * (tmp / self.nsamples)

    def free(self):
        if DEBUG:
            self.inp1 = None
            self.out1 = None
        self.mask = None
        self.Losses = None
        self.Trace = None
        torch.cuda.empty_cache()

    def fasterquant(
        self,
        blocksize=128,
        percdamp=0.01,
        groupsize=-1,
        actorder=False,
        static_groups=False,
        include_m_step=False,
        use_vq=False,
        svd_rank=None,
        hessian_weighted_lookups=False,
        only_init_kmeans=False,
        use_incoherent=False,
        incoherent_param=10,
        logger=None,
        args=None,
        coherent_type="incoherent",
        name="layer",
    ):
        W = self.layer.weight.data.clone()  # o i
        W = W.float()

        self.mask = self.mask * 1e3

        if False:  # use_incoherent:
            # molan = get_moran_stat(W.cpu().numpy())
            # print("Moran I:", molan)

            if coherent_type == "incoherent":
                assert incoherent_param > 0
                # U, s, Vh = torch.linalg.svd(H)
                is_incoherent_number = W.abs().max() * (W.abs() ** 2).sum().mean().rsqrt() * np.sqrt(W.shape[0] * W.shape[1])
                is_incoherent = is_incoherent_number < incoherent_param

                # 决定是否使用vq
                if not is_incoherent:
                    # draw_img(W.detach().cpu().numpy(), "experiment/a_imgs/incoherent", name+".png")
                    if use_vq:
                        use_vq = False
                        include_m_step = False
                        # print("weight is incoherent use m_quant instead of vq_quant")
                        logger.info(f"weight is incoherent use m_quant instead of vq_quant")
                    else:
                        use_vq = True
                        include_m_step = True
                        # print("weight is coherent use vq_quant instead of mquant")
                        logger.info(f"weight is coherent use vq_quant instead of mquant")
                    self.quantizer = self.quantizer_bak
                else:
                    # print(f"weight is coherent with u={incoherent_param} and incoherent_number {is_incoherent_number}")
                    logger.info(f"weight is coherent with u={incoherent_param} and incoherent_number {is_incoherent_number}")
                    # draw_img(W.detach().cpu().numpy(), "experiment/a_imgs/coherent", name+".png")
                    # if is_incoherent_number < 5: # 降低一bit
                    #     self.quantizer.configure(args.wbits-1, perchannel=True, sym=args.sym, mse=False)
                    #     logger.info(f"weight is very coherent {is_incoherent_number}")
            else:
                raise NotImplementedError

        self.tick = time.time()

        if not self.quantizer.ready() and not use_vq:
            self.quantizer.find_params(W, weight=True)

        vq_dim = self.assignments = None
        self.assignments = []

        if use_vq and hasattr(self.quantizer, "vq_dim"):
            vq_dim = self.quantizer.vq_dim  # 1
            groupsize = self.quantizer.get_groupsize(W, groupsize)  # 256
            assert blocksize % vq_dim == 0
            print(f"weight shape is {W.shape}, groupsize is {groupsize}")

        Losses = torch.zeros_like(W)
        Q = torch.zeros_like(W)

        mq_count = 0
        vq_count = 0
        mquantizers = []

        for i1 in range(0, self.columns, groupsize):
            i2 = min(i1 + groupsize, self.columns)
            W1 = W[..., i1:i2]

            if use_incoherent and args.reset_vq:  # old is not open
                use_vq = True  # xc

            if use_vq:
                self.assignments.append([])

            # extra_args = {}
            if args.mask_q:
                extra_args = {"mask": self.mask[..., i1:i2]}
            else:
                extra_args = {}

            if not use_vq or not hasattr(self.quantizer, "find_params_mul") or not args.mask_q:
                dist = self.quantizer.find_params(W1, weight=True, **extra_args)
            else:
                dist = self.quantizer.find_params_mul(W1, weight=True, **extra_args)

            if False:
                if coherent_type == "incoherent":
                    assert incoherent_param > 0
                    # U, s, Vh = torch.linalg.svd(H)
                    is_incoherent_number = W_group_scaled.abs().max() * (W_group_scaled.abs() ** 2).sum().mean().rsqrt() * np.sqrt(W_group_scaled.shape[0] * W_group_scaled.shape[1])

                    is_incoherent_number = is_incoherent_number + dist * 1e3  # / 2
                    # print(is_incoherent_number)

                    is_incoherent = is_incoherent_number < incoherent_param

                    if not is_incoherent:
                        # draw_img(W_group_scaled.detach().cpu().numpy(), "experiment/a_imgs/incoherent_block", name+f"_{i1 + i}.png")
                        mq_count += 1
                        if use_vq:
                            use_vq = False
                            include_m_step = False
                            # print("weight is incoherent use m_quant instead of vq_quant")
                            logger.info(f"weight is incoherent use m_quant instead of vq_quant")
                        else:
                            use_vq = True
                            include_m_step = True
                            # print("weight is coherent use vq_quant instead of mquant")
                            logger.info(f"weight is coherent use vq_quant instead of mquant")

                        id = i1 + i
                        mquantizer = Quantizer()
                        mquantizer.configure(args.wbits, perchannel=True, sym=args.sym, mse=args.mq_mse)
                        mquantizer.find_params(W_group_scaled, weight=True, **extra_args)
                        # mquantizer.optimize_s()

                        mquantizers.append(((id, id + groupsize), mquantizer))
                        self.quantizer_bak = mquantizer

                    else:
                        mquantizers.append([])
                        # print(f"weight is coherent with u={incoherent_param} and incoherent_number {is_incoherent_number}")
                        logger.info(f"weight is coherent with u={incoherent_param} and incoherent_number {is_incoherent_number}")
                        # draw_img(W_group_scaled.detach().cpu().numpy(), "experiment/a_imgs/coherent_block", name+f"_{i1 + i}.png")
                        vq_count += 1
                        # if is_incoherent_number < 5: # 降低一bit
                        #     self.quantizer.configure(args.wbits-1, perchannel=True, sym=args.sym, mse=False)
                        #     logger.info(f"weight is very coherent {is_incoherent_number}")
                else:
                    raise NotImplementedError

            if not use_vq or not hasattr(self.quantizer, "find_params_mul") or not args.mask_q:
                q = quantize(
                    W1,
                    self.quantizer.scale,
                    self.quantizer.zero,
                    self.quantizer.maxq,
                )

                Q[..., i1:i2] = q
                Losses[..., i1:i2] = (W1 - q) ** 2

            else:
                q, assmt = vq_quantize_mul(W1, self.quantizer, mask=None)

                self.assignments[-1].append(assmt)

                Q[..., i1:i2] = q
                Losses[..., i1:i2] = (W1 - q) ** 2  # R x D / 1 x D

        torch.cuda.synchronize()
        print("time %.2f" % (time.time() - self.tick))
        # print("error", torch.sum(Losses).item())
        logger.info(f"error:{torch.sum(Losses).item()}")

        if mq_count:
            logger.info(f"mq percent :{ mq_count / (mq_count + vq_count)}")

        # if include_m_step: # and not use_incoherent  [768, 64]
        #     Q = self.lut_m_step(Q, groupsize, self.quantizer, scale=S, svd_rank=svd_rank, logger=logger, mquantizers=mquantizers)

        self.layer.weight.data = Q.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)


class AWQ:
    def __init__(self, layer):
        self.layer = layer
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        # print(W.shape)
        if len(W.shape) == 3:
            self.rows = W.shape[1]  # H
            self.columns = W.shape[2]  # W
        else:
            self.rows = W.shape[0]  # H
            self.columns = W.shape[1]  # W
        self.nsamples = 0
        self.mask = None
        self.input_data = None
        self.recorded = True

    def add_batch(self, inp, out):
        if self.recorded:
            add_extr_flops(inp.numel())
            self.recorded = False

        if DEBUG:
            self.inp1 = inp
            self.out1 = out

        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)

        self.input_data = inp
        # self.input_data.append( inp.unsqueeze(0) )

    def free(self):
        if DEBUG:
            self.inp1 = None
            self.out1 = None
        self.mask = None
        self.Losses = None
        self.Trace = None
        torch.cuda.empty_cache()

    def serach_scale(self, W):
        with torch.no_grad():
            org_out = self.layer(self.input_data)
            if isinstance(org_out, tuple):
                org_out = org_out[0]

        best_error = float("inf")
        best_ratio = -1
        best_scales = None

        n_grid = 20
        history = []

        if len(self.input_data.shape) == 3:
            inp = self.input_data.abs().mean(0).mean(0)
        else:
            assert len(self.input_data.shape) == 2
            inp = self.input_data.abs().mean(0)

        for ratio in range(1, n_grid):
            ratio = ratio * 1 / n_grid
            scales = inp.pow(ratio).clamp(min=1e-4).view(-1)
            scales = scales / (scales.max() * scales.min()).sqrt()

            W_r = W * scales.view(1, -1).to(W.device)
            assert W_r.shape == W.shape

            self.quantizer.find_params(W_r, weight=True)
            self.layer.weight.data = (self.quantizer.quantize(W_r) / scales.view(1, -1).to(W.device)).to(self.layer.weight.data.dtype)

            out = self.layer(self.input_data)
            if isinstance(out, tuple):
                out = out[0]

            loss = (org_out - out).float().pow(2).mean().item()  # float prevents overflow
            is_best = loss < best_error
            if is_best:
                history.append(loss)
                best_error = loss
                best_ratio = ratio
                best_scales = scales

        print(best_error)
        self.layer.weight.data = (W * best_scales.view(1, -1).to(W.device)).to(self.layer.weight.data.dtype)
        return best_scales.view(1, -1).to(W.device)

    def fasterquant(
        self,
        blocksize=128,
        percdamp=0.01,
        groupsize=-1,
        actorder=False,
        static_groups=False,
        include_m_step=False,
        use_vq=False,
        svd_rank=None,
        hessian_weighted_lookups=False,
        only_init_kmeans=False,
        use_incoherent=False,
        incoherent_param=10,
        logger=None,
        args=None,
        coherent_type="incoherent",
        name="layer",
    ):
        W = self.layer.weight.data.clone()  # o i
        W = W.float()

        self.tick = time.time()

        if not self.quantizer.ready() and not use_vq:
            self.quantizer.find_params(W, weight=True)

        # Losses = torch.zeros_like(W)
        Q = torch.zeros_like(W)

        scale = self.serach_scale(W)
        W = W * scale

        # flops_number += self.Q.numel()*torch.log(self.Q.numel())

        for i1 in range(0, self.columns, groupsize):
            i2 = min(i1 + groupsize, self.columns)
            W1 = W[..., i1:i2]

            dist = self.quantizer.find_params(W1, weight=True)
            q = quantize(
                W1,  # 64
                self.quantizer.scale,  # 64
                self.quantizer.zero,
                self.quantizer.maxq,
            )
            Q[..., i1:i2] = q
            # Losses[..., i1:i2] = (W1 - q) ** 2  # R x D / 1 x D

        torch.cuda.synchronize()
        # print("time %.2f" % (time.time() - self.tick))
        # print("error", torch.sum(Losses).item())
        # logger.info(f"error:{torch.sum(Losses).item()}")

        self.layer.weight.data = (Q / scale).reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)


class Qurot_Module(torch.nn.Module):
    def __init__(self, ori_m, Q):
        super().__init__()
        self.Q = Q
        self.ori_m = ori_m

    def forward(self, x):
        ori_dtype = x.dtype
        out = torch.matmul(self.ori_m(x).to(torch.float64), self.Q.T).to(ori_dtype)
        # out = self.ori_m(x)
        return out


class Qurot:
    def __init__(self, layer):
        self.layer = layer
        self.dev = self.layer.weight.device

        Q = get_orthogonal_matrix(self.layer.weight.shape[0], mode="hadamard")
        self.Q = Q
        dtype = self.layer.weight.data.dtype
        W1 = self.layer.weight.data.to(dtype=torch.float64)
        self.layer.weight.data = torch.matmul(Q.T, W1).to(dtype=dtype)
        # if hasattr(self.layer, "bias") and self.layer.bias is not None:
        #     bias = self.layer.bias.data.to(torch.float64)
        #     self.layer.bias.data = torch.matmul(bias, self.Q).to(dtype=dtype)
        # quant
        add_extr_flops(self.Q.shape[0] * torch.log(torch.tensor(self.Q.shape[0])))

        W = layer.weight.data.clone()
        # print(W.shape)
        if len(W.shape) == 3:
            self.rows = W.shape[1]  # H
            self.columns = W.shape[2]  # W
        else:
            self.rows = W.shape[0]  # H
            self.columns = W.shape[1]  # W
        self.nsamples = 0
        self.mask = None
        self.input_data = None

    def add_batch(self, inp, out):
        if DEBUG:
            self.inp1 = inp
            self.out1 = out

    def free(self):
        if DEBUG:
            self.inp1 = None
            self.out1 = None
        self.mask = None
        self.Losses = None
        self.Trace = None
        self.Q = None
        torch.cuda.empty_cache()

    def fasterquant(
        self,
        blocksize=128,
        percdamp=0.01,
        groupsize=-1,
        actorder=False,
        static_groups=False,
        include_m_step=False,
        use_vq=False,
        svd_rank=None,
        hessian_weighted_lookups=False,
        only_init_kmeans=False,
        use_incoherent=False,
        incoherent_param=10,
        logger=None,
        args=None,
        coherent_type="incoherent",
        name="layer",
    ):
        W = self.layer.weight.data.clone()  # o i
        W = W.float()

        self.tick = time.time()

        if not self.quantizer.ready() and not use_vq:
            self.quantizer.find_params(W, weight=True)

        # Losses = torch.zeros_like(W)
        Q_w = torch.zeros_like(W)

        for i1 in range(0, self.columns, groupsize):
            i2 = min(i1 + groupsize, self.columns)
            W1 = W[..., i1:i2]

            dist = self.quantizer.find_params(W1, weight=True)
            q = quantize(
                W1,  # 64
                self.quantizer.scale,  # 64
                self.quantizer.zero,
                self.quantizer.maxq,
            )
            Q_w[..., i1:i2] = q
            # Losses[..., i1:i2] = (W1 - q) ** 2  # R x D / 1 x D

        torch.cuda.synchronize()
        self.layer.weight.data = Q_w.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)

        W = self.layer.weight.data.to(dtype=torch.float64)
        self.layer.weight.data = torch.matmul(self.Q, W).to(dtype=self.layer.weight.data.dtype)
        # self.layer = Qurot_Module(self.layer, self.Q)


class RTN:
    def __init__(self, layer):
        self.layer = layer
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        # print(W.shape)
        if len(W.shape) == 3:
            self.rows = W.shape[1]  # H
            self.columns = W.shape[2]  # W
        else:
            self.rows = W.shape[0]  # H
            self.columns = W.shape[1]  # W
        self.nsamples = 0
        self.mask = None

        self.input_data = None

    def add_batch(self, inp, out):
        if DEBUG:
            self.inp1 = inp
            self.out1 = out

        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)

        self.input_data = inp
        # self.input_data.append( inp.unsqueeze(0) )

    def free(self):
        if DEBUG:
            self.inp1 = None
            self.out1 = None
        self.mask = None
        self.Losses = None
        self.Trace = None
        torch.cuda.empty_cache()

    def fasterquant(
        self,
        blocksize=128,
        percdamp=0.01,
        groupsize=-1,
        actorder=False,
        static_groups=False,
        include_m_step=False,
        use_vq=False,
        svd_rank=None,
        hessian_weighted_lookups=False,
        only_init_kmeans=False,
        use_incoherent=False,
        incoherent_param=10,
        logger=None,
        args=None,
        coherent_type="incoherent",
        name="layer",
    ):
        W = self.layer.weight.data.clone()  # o i
        W = W.float()

        self.tick = time.time()

        if not self.quantizer.ready() and not use_vq:
            self.quantizer.find_params(W, weight=True)

        # Losses = torch.zeros_like(W)
        Q = torch.zeros_like(W)

        for i1 in range(0, self.columns, groupsize):
            i2 = min(i1 + groupsize, self.columns)
            W1 = W[..., i1:i2]

            dist = self.quantizer.find_params(W1, weight=True)
            q = quantize(
                W1,  # 64
                self.quantizer.scale,  # 64
                self.quantizer.zero,
                self.quantizer.maxq,
            )
            Q[..., i1:i2] = q
            # Losses[..., i1:i2] = (W1 - q) ** 2  # R x D / 1 x D

        torch.cuda.synchronize()
        # print("time %.2f" % (time.time() - self.tick))
        # print("error", torch.sum(Losses).item())
        # logger.info(f"error:{torch.sum(Losses).item()}")

        self.layer.weight.data = (Q).reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)


class Kmeans:
    def __init__(self, layer):
        self.layer = layer
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.rows = W.shape[0]  # H
        self.columns = W.shape[1]  # W

    def add_batch(self, inp, out):
        if DEBUG:
            self.inp1 = inp
            self.out1 = out

    def fasterquant(
        self,
        blocksize=128,
        percdamp=0.01,
        groupsize=-1,
        actorder=False,
        static_groups=False,
        include_m_step=False,
        use_vq=False,
        svd_rank=None,
        hessian_weighted_lookups=False,
        only_init_kmeans=False,
        use_incoherent=False,
        incoherent_param=10,
        logger=None,
        args=None,
        coherent_type="incoherent",
        name="layer",
    ):
        W = self.layer.weight.data.clone()  # o i
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float()

        self.tick = time.time()

        if not self.quantizer.ready() and not use_vq:
            self.quantizer.find_params(W, weight=True)

        vq_dim = self.assignments = None
        S = vq_scaling_blocksize = vq_scaling_n_bits = None
        if use_vq:
            vq_dim = self.quantizer.vq_dim
            groupsize = self.quantizer.get_groupsize(W, groupsize)
            self.assignments = []
            assert blocksize % vq_dim == 0

            vq_scaling_blocksize = self.quantizer.vq_scaling_blocksize
            vq_scaling_n_bits = self.quantizer.vq_scaling_n_bits
            if vq_scaling_blocksize > 0:
                assert vq_scaling_blocksize % vq_dim == 0
                S = torch.ones_like(W)

            print(f"VQ scaling BS {vq_scaling_blocksize} @ {vq_scaling_n_bits}b " f"({self.quantizer.vq_scaling_domain} domain)")
            print(f"Using Hessian-aware K-means {hessian_weighted_lookups}")

        Losses = torch.zeros_like(W)
        Q = torch.zeros_like(W)

        mquantizers = []
        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            if use_vq and vq_scaling_blocksize > 0:
                W1_scaled, S1 = self.quantizer.blockwise_normalize_data(
                    W1,
                    vq_scaling_blocksize,
                    self.quantizer.vq_scaling_norm,
                    vq_scaling_n_bits,
                    self.quantizer.vq_scaling_domain,
                )
                S[:, i1:i2] = S1
            else:
                W1_scaled = W1
                S1 = torch.ones_like(W1)

            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)

            for i in range(count):
                if groupsize != -1:
                    if (i1 + i) % groupsize == 0:
                        extra_args = {}
                        W_group = W[:, (i1 + i) : (i1 + i + groupsize)]
                        W_group_scaled = W_group

                        if use_vq:
                            self.assignments.append([])
                            if vq_scaling_blocksize > 0:
                                assert vq_scaling_blocksize % vq_dim == 0
                                W_group_scaled, S_group = self.quantizer.blockwise_normalize_data(
                                    W_group,
                                    vq_scaling_blocksize,
                                    self.quantizer.vq_scaling_norm,
                                    self.quantizer.vq_scaling_n_bits,
                                    self.quantizer.vq_scaling_domain,
                                )

                        dist = self.quantizer.find_params(W_group_scaled, weight=True, **extra_args)

                if i % vq_dim == 0:
                    w = W1[:, i : i + vq_dim]  # R x D  4096*1
                    w_scaled = W1_scaled[:, i : i + vq_dim]  # R x D
                    s = S1[:, i : i + vq_dim]

                    H_inv_diag = None
                    q, assmt = vq_quantize(w_scaled, self.quantizer, H_inv_diag=H_inv_diag)  # R x 1 x D, R x 1

                    self.assignments[-1].append(assmt)
                    Q1[:, i : i + vq_dim] = q

            Q[:, i1:i2] = Q1

        torch.cuda.synchronize()
        print("time %.2f" % (time.time() - self.tick))
        # print("error", torch.sum(Losses).item())
        logger.info(f"error:{torch.sum(Losses).item()}")

        if isinstance(self.layer, transformers.Conv1D):
            Q = Q.t()

        self.layer.weight.data = Q.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)

    def free(self):
        if DEBUG:
            self.inp1 = None
            self.out1 = None
        self.H = None
        self.Losses = None
        self.Trace = None
        torch.cuda.empty_cache()
