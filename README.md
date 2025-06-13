# RWKVQuant: Quantizing the RWKV Family with Proxy Guided Hybrid of Uniform and Vector Quantization

📄 [**Paper on arXiv**](https://arxiv.org/abs/2505.03803)

---

## 🧱 1. Setup & Installation

```bash
conda env create -f environment.yml

mkdir dev && cd dev
git clone https://github.com/Dao-AILab/fast-hadamard-transform.git
pip instal -e .

```

---

## 📥 2. Download Models

```bash
cd ckpt
pip install huggingface_hub
python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='BlinkDL/rwkv-7-pile', filename='RWKV-x070-Pile-168M-20241120-ctx4096.pth', local_dir='./rwkv_model')"
```

## 🏋️ QUANT

```bash
python rwkv_7.py ckpt/RWKV-x070-Pile-168M-20241120-ctx4096.pth --columns-per-group 32 --use-vq --kmeans-iters 100 --kmeans-init-method mahalanobis --hessian-weighted-lookups --include-m-step --wbits 3 --vq-dim 1 --groupsize 64 --use_incoherent --incoherent_param 8 --codebook-bitwidth 8 --svd-rank 0.5 --reset_vq --mq_mse --mask_q --quant_mul --quant_conv --k_ablition 4 --coherent_type hce_v2 --incoherent_param 1.7781 135.5616 _6_older.py  $model_14B_path --wbits 3 --groupsize 32  --svd-rank 0.5 --quant_mul --quant_conv --use_qurot --coherent_type "mad"
```