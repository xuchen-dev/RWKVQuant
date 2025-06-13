import time
import os
import time
import torch
import torch.nn as nn
from torch.profiler import profile, record_function, ProfilerActivity
from quant.gptq import *
from utils.modelutils import *
from quant import *
from quant.vq_quant import *
from models.rwkv_v6 import get_mod_rwkv_model, get_rwkv_dataset, get_rwkv_token, eval_lambada
from loguru import logger
from quant.utils import seed_everything
from utils.vptq_utils.quantizer import NPVectorQuantizer
from utils.vptq_utils.vptq import VPTQ
from quant.quant import Quantizer, quantize


@torch.no_grad()
def quant_rwkv(model, dataloader, dev, args):
    print("Starting ...")

    layers = model.blocks
    rkwv_tokenizer = get_rwkv_token()

    inps = [[] for _ in range(args.nsamples)]
    cache = {"i": 0}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp):
            inps[cache["i"]] = inp
            cache["i"] += 1
            raise ValueError

    for batch in dataloader:
        try:
            src = [0] + rkwv_tokenizer.encode(batch[0])
            dst = rkwv_tokenizer.encode(batch[1])
            inp_emb = torch.tensor(src + dst).reshape(1, -1).cuda()
            model(inp_emb[:, :1].to(dev))

            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                on_trace_ready=torch.profiler.tensorboard_trace_handler("./log"),
                record_shapes=True,
                with_stack=True,
                with_flops=True,
            ) as prof:
                with record_function("model_inference"):
                    model(inp_emb)

            print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
            print(prof.key_averages(group_by_input_shape=True).table())

        except ValueError:
            raise NotImplementedError

    layers[0] = layers[0].module
    torch.cuda.empty_cache()
    outs = [[] for _ in range(args.nsamples)]

    if args.use_vq:
        QClass = lambda: VQQuantizer(
            vq_dim=args.vq_dim,
            columns_per_group=args.columns_per_group,
            vq_scaling_blocksize=args.vq_scaling_blocksize,
            vq_scaling_norm=args.vq_scaling_norm,
            vq_scaling_n_bits=args.vq_scaling_n_bits,
            vq_scaling_domain=args.vq_scaling_domain,
            kmeans_init_method=args.kmeans_init_method,
            assignment_chunk_size=args.assignment_chunk_size,
            kmeans_iters=args.kmeans_iters,
            codebook_bitwidth=args.codebook_bitwidth,
            quantize_per_codebook=args.quantize_per_codebook,
            quantize_during_kmeans=args.quantize_during_kmeans,
            n_subsample=args.kpp_n_subsample,
        )
        QClass_bak = Quantizer
    elif args.use_vptq:
        QClass = lambda: NPVectorQuantizer(
            vector_lens=[-1, 8],
            num_centroids=[-1, 256],
            num_res_centroids=[-1, 256],
            npercent=0,
            group_size=args.groupsize,
            kmeans_mode="hessian",
            iter=args.kmeans_iters,
            debug=True,
            logger=logger,
        )
        QClass_bak = None
    else:
        QClass = Quantizer
        QClass_bak = None

    print("Ready.")

    quantizers = {}
    for i in range(len(layers)):
        layer = layers[i]
        full = find_layers(
            layer,
            find_mul=args.quant_mul,
            find_conv=args.quant_conv,
        )

        sequential = [[k for k in list(full.keys()) if "block_sparse_moe.gate" not in k]]

        for names in sequential:
            subset = {n: full[n] for n in names}

            gptq = {}
            for name in subset:
                if "mul" in name:
                    gptq[name] = Mask_Q(subset[name])
                else:
                    if args.use_kmeans:
                        gptq[name] = Kmeans(subset[name])
                    elif args.use_rtn:
                        gptq[name] = RTN(subset[name])
                    elif args.use_awq:
                        gptq[name] = AWQ(subset[name])
                    elif args.use_vptq:
                        gptq[name] = VPTQ(subset[name])
                    elif args.use_qurot:
                        gptq[name] = Qurot(subset[name])
                        QClass = Quantizer
                    else:
                        gptq[name] = GPTQ(subset[name])

                if not args.mask_q:
                    gptq[name].quantizer = Quantizer
                else:
                    gptq[name].quantizer = QClass()

                if args.use_vptq:
                    if subset[name].weight.shape[0] >= subset[name].weight.shape[1]:
                        gptq[name].quantizer.set_num_centroids([-1, 2**10])

                gptq[name].quantizer.configure(args.wbits, perchannel=True, sym=args.sym, mse=False)
                # for incoherent
                if QClass_bak is not None:
                    gptq[name].quantizer_bak = QClass_bak()
                    gptq[name].quantizer_bak.configure(args.wbits, perchannel=True, sym=args.sym, mse=False)

            def add_batch(name):
                def tmp(_, inp, out):
                    gptq[name].add_batch(inp[0].data, out.data)

                return tmp

            handles = []
            for name in subset:
                handles.append(subset[name].register_forward_hook(add_batch(name)))
            for j in range(args.nsamples):
                outs[j] = layer(inps[j])
            for h in handles:
                h.remove()

            for name in subset:
                # print(i, name)
                logger.info(f"{i}_{name}")
                print("Quantizing ...")
                gptq[name].fasterquant(
                    percdamp=args.percdamp,
                    groupsize=args.groupsize,
                    actorder=args.act_order,
                    include_m_step=args.include_m_step,
                    use_vq=args.use_vq,
                    svd_rank=args.svd_rank,
                    hessian_weighted_lookups=args.hessian_weighted_lookups,
                    only_init_kmeans=args.only_init_kmeans,
                    use_incoherent=args.use_incoherent,
                    incoherent_param=args.incoherent_param,
                    logger=logger,
                    args=args,
                    name=str(i) + name,
                )
                quantizers["model.layers.%d.%s" % (i, name)] = gptq[name].quantizer
                gptq[name].free()

        for j in range(args.nsamples):
            outs[j] = layer(inps[j])

        # layers[i] = layer.cpu()
        del layer
        del gptq
        torch.cuda.empty_cache()
        inps, outs = outs, inps
    return quantizers


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "model",
        type=str,
        help="rwkv model to load",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed for sampling the calibration data.")
    parser.add_argument("--nsamples", type=int, default=128, help="Number of calibration data samples.")
    parser.add_argument("--use_incoherent", action="store_true", help="Whether to use_incoherent.")
    parser.add_argument("--incoherent_param", type=float, default=10.0, help="Number of incoherent.")
    parser.add_argument("--rotate", action="store_true", help="Whether to rotate.")
    parser.add_argument("--reset_vq", action="store_true", help="Whether to rotate.")
    parser.add_argument("--use_vptq", action="store_true", help="Whether to use_vptq.")
    parser.add_argument("--mq_mse", action="store_true", help="Whether to mq_mse.")
    parser.add_argument("--use_awq", action="store_true", help="Whether to use_awq.")
    parser.add_argument("--mask_q", action="store_true", help="Whether to mask_q.")
    parser.add_argument("--quant_mul", action="store_true", help="Whether to quant_mul.")
    parser.add_argument("--quant_conv", action="store_true", help="Whether to quant_conv.")
    parser.add_argument("--use_qurot", action="store_true", help="Whether to use_qurot.")
    parser.add_argument("--coherent_type", type=str, default="default", help="Choose method for calib coherent")
    parser.add_argument(
        "--percdamp",
        type=float,
        default=0.01,
        help="Percent of the average Hessian diagonal to use for dampening.",
    )
    parser.add_argument("--nearest", action="store_true", help="Whether to run the RTN baseline.")
    parser.add_argument(
        "--wbits",
        type=float,
        default=16,
        help="#bits to use for quantization; use 16 for evaluating base model.",
    )
    parser.add_argument("--no-quant", action="store_true", help="If set, run FP16 model without quantization")
    parser.add_argument(
        "--groupsize",
        type=int,
        default=-1,
        help="Groupsize to use for quantization; default uses full row.",
    )
    parser.add_argument("--sym", action="store_true", help="Whether to perform symmetric quantization.")
    parser.add_argument("--save", action="store_true", help="Save quantized checkpoint")
    parser.add_argument(
        "--act-order",
        action="store_true",
        help="Whether to apply the activation order GPTQ heuristic",
    )
    parser.add_argument("--use-vq", action="store_true", help="If set, use VQ (multi-dim non-uniform) quantization")
    parser.add_argument("--vq-dim", type=int, default=2, help="Dimensionality of VQ (if using)")
    parser.add_argument("--vq-scaling-blocksize", type=int, default=-1, help="VQ scaling block size")

    parser.add_argument("--vq-scaling-n-bits", type=int, default=4, help="VQ scaling bit-width")

    parser.add_argument("--vq-scaling-norm", type=str, default="max", help="VQ scaling norm")
    parser.add_argument(
        "--vq-scaling-domain",
        type=str,
        default="log",
        choices=["log", "linear"],
        help="VQ scaling domain",
    )

    parser.add_argument(
        "--include-m-step",
        action="store_true",
        help="If set, perform an M-step (centroid updating) after GPTQ with VQ",
    )
    parser.add_argument(
        "--columns-per-group",
        type=int,
        default=None,
        help="For group-/blockwise quant: force number of columns each group spans (rest is absorbed in rows)",
    )
    parser.add_argument(
        "--kmeans-init-method",
        type=str,
        default="cdf",
        choices=["cdf", "kpp", "mahalanobis"],
        help="init method for Kmeans",
    )
    parser.add_argument(
        "--assignment-chunk-size",
        type=int,
        default=None,
        help="Chunk assignment step for better memory management",
    )
    parser.add_argument("--kmeans-iters", type=int, default=10)
    parser.add_argument("--codebook-bitwidth", type=int, default=None, help="Bitwidth for codebook quantization")
    parser.add_argument(
        "--quantize-per-codebook",
        action="store_true",
        default=False,
        help="Quantize codebooks individually (more overhead) or per column block",
    )
    parser.add_argument(
        "--quantize-during-kmeans",
        action="store_true",
        default=False,
        help="Quantize codebooks after every M-step. If not set: only quantize after k-means",
    )
    parser.add_argument(
        "--model-type",
        choices=["llama", "mistral", "mixtral"],
        default="llama",
        help="In case this is a Mistral model (GPTQ layerwise remains the same)",
    )
    parser.add_argument("--kpp-n-subsample", type=int, default=10000)
    parser.add_argument("--svd-rank", type=float, default=None)
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to save model in")
    parser.add_argument("--hessian-weighted-lookups", action="store_true", default=False)
    parser.add_argument("--only-init-kmeans", action="store_true", default=False)
    parser.add_argument("--over_all_eval", action="store_true", default=False)

    args = parser.parse_args()
    seed_everything(args.seed)

    name, path = get_task_name(args)
    logger.add(f"{path}/{time.localtime()[1]}_{time.localtime()[2]}_{time.localtime()[3]}_{time.localtime()[4]}.log", format="{time} {level} {message}", level="INFO")
    logger.info(args)

    if not args.use_vq:
        args.wbits = int(args.wbits)

    model = get_mod_rwkv_model(
        args.model,
    )
    mdoel = model.to(DEV)
    model.eval()

    dataloader = get_rwkv_dataset(args.nsamples)

    # import torch
    # import torch.nn as nn
    # from torch.profiler import profile, record_function, ProfilerActivity

    # dummy_input = torch.randn(1,1,4096).to(torch.int32).cuda()  # Batch size 1, 3 channels, 8x8 image

    # # 使用 torch.profiler 统计计算量
    # with profile(
    #     activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],  # 支持 CPU 和 CUDA
    #     on_trace_ready=torch.profiler.tensorboard_trace_handler('./log'),  # 保存数据以便于可视化
    #     record_shapes=True,  # 记录张量形状
    #     with_stack=True  # 可选，记录调用堆栈
    # ) as prof:
    #     with record_function("model_inference"):  # 添加自定义标签
    #         model(dummy_input)

    # # 打印统计结果
    # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    # # 如果需要 FLOPs，可以通过以下方式
    # print(prof.key_averages(group_by_input_shape=True).table())

    with torch.no_grad():
        if args.wbits < 16 and not args.nearest and not args.no_quant:
            tick = time.time()
            quantizers = quant_rwkv(model, dataloader, DEV, args)
            print(time.time() - tick)

    tokenizer = get_rwkv_token()
    dataloader = get_rwkv_dataset()
    eval_lambada(dataloader, model, tokenizer, logger)

    if path is not None and args.save:
        output_path = os.path.join(path, f"{name}_rwkv6.pt")
        torch.save(model, output_path)

    if args.over_all_eval:
        import subprocess

        subprocess.run(
            ["/data01/home/xuchen/miniconda3/envs/eagle/bin/python", "eval/run_lm_eval_6.py", "--model", f"experiment/latest/{name}/{name}_rwkv6.pt", "--output_dir", f"experiment/latest/{name}"]
        )
