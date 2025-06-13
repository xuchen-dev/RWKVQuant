# RWKVQuant: Quantizing the RWKV Family with Proxy Guided Hybrid of Uniform and Vector Quantization

📄 [**Paper on arXiv**](https://arxiv.org/abs/2505.03803)

---

## 🧱 1. Setup & Installation

```bash
conda env create -f environment.yml
conda activate rwkv

mkdir dev && cd dev
git clone https://github.com/Dao-AILab/fast-hadamard-transform.git
cd fast-hadamard-transform && pip install -e .

```

---

## 📥 2. Download Models

```bash
cd ckpt && rm -r RWKV-x070-Pile-168M-20241120-ctx4096.pth
python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='BlinkDL/rwkv-7-pile', filename='RWKV-x070-Pile-168M-20241120-ctx4096.pth', local_dir='.')"
```

## 🏋️ QUANT
Run the command in the home directory:
```bash
python rwkv_7.py ckpt/RWKV-x070-Pile-168M-20241120-ctx4096.pth --columns-per-group 32 --use-vq --kmeans-iters 100 --kmeans-init-method mahalanobis --hessian-weighted-lookups --include-m-step --wbits 3 --vq-dim 1 --groupsize 64 --use_incoherent --incoherent_param 8 --codebook-bitwidth 8 --svd-rank 0.5 --reset_vq --mq_mse --mask_q --quant_mul --quant_conv
```

## Feedback
If you have any questions or ideas, please feel free to contact us via email `chen.xu@houmo.ai` or Wechat ( `C-C_wechat` or QR code below ). Thank you for your attention to this repository and RWKV, and we appreciate your valuable feedback.

[![Wechat](docs/wechat.png)]


## Acknowledgement
We would like to express our heartfelt gratitude to the authors and staff of RWKV. They have made remarkable contributions to the development of LLMs, lightweight LLMs, and long-text conversations. 
* `https://github.com/BlinkDL/RWKV-LM` 
* `https://huggingface.co/BlinkDL`