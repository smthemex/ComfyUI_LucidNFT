# ComfyUI_LucidNFT

[LucidNFT](https://github.com/W2GenAI-Lab/LucidNFT)：LR-Anchored Multi-Reward Preference Optimization for Generative Real-World Super-Resolution

# Update

* 改成comfyUI的核心封装，提升推理速度/内存占用/changed to comfyUI core wrapper to improve inference speed/memory usage
* 记得给官方[LucidNFT](https://github.com/W2GenAI-Lab/LucidNFT)点星;  Remember to give the official [LucidNFT](https://github.com/W2GenAI-Lab/LucidNFT) a star

---

## 1. 安装

`custom_nodes/` 下：

```
git clone https://github.com/smthemex/ComfyUI_LucidNFT
```

## 2. 权重

```
ComfyUI/models/
├── diffusion_models/flux1-dev-fp8.safetensors        # 任意标准 flux DiT（推荐 fp8）
├── vae/UltraFlux.safetensors                        # 可选：出 2K 时用（F16 /16）
├── lora/LucidNFT_lora_dit.safetensors  
├── clip_vision/siglip2-so400m-patch16-512.safetensors  # 建议改个名，否则都叫 model.safetensors
└── LucidFlux/                                       # 与 ComfyUI_LucidFlux 共用同一目录
    ├── general_swinir_v1.ckpt                       # DiffBIR 先验
    ├── lucidflux.pth                                # 双条件分支（condition_lq/ldr + modulation）
    ├── prompt_embeddings.pt                         # 官方预计算文本 embedding（txt 512×4096 + vec 768）
    ├── lucid_connector.pth                          # SigLIP→4096 的 Redux 映射（也可直接给 lucidflux.pth）
    └── lora_condition/{adapter_config.json, adapter_model.safetensors}  # LucidNFT 的条件分支 LoRA（PEFT 原始格式）

转换后（跑一次 tools/convert_dit_lora.py 得到，见第 3 节）：
models/loras/LucidNFT_lora_dit.safetensors   # lora_dit 转成的标准 comfy LoRA
```

下载：`lucidflux.pth` / `prompt_embeddings.pt` → [W2GenAI/LucidFlux](https://huggingface.co/W2GenAI/LucidFlux/tree/main)；
`lora_condition` → [W2GenAI/LucidNFT](https://huggingface.co/W2GenAI/LucidNFT/tree/main)；
`lora_dit` → [smthemex/UltraFlux-v1-gguf](https://huggingface.co/smthemex/UltraFlux-v1-gguf)（转化后的comfyUI lora）；
SigLIP → [google/siglip2-so400m-patch16-512](https://huggingface.co/google/siglip2-so400m-patch16-512/tree/main)（只下单体模型）；
SwinIR → [lxq007/DiffBIR](https://huggingface.co/lxq007/DiffBIR/tree/main) 的 `general_swinir_v1.ckpt`；
connector → [smthem/LucidFLUX-connector](https://huggingface.co/smthem/LucidFLUX-connector)。

# 3 .Example
![](https://github.com/smthemex/ComfyUI_LucidNFT/blob/main/example_workflows/example.png)

## 4. Citation

```
@article{fei2026lucidnft,
  title={LucidNFT: LR-Anchored Multi-Reward Preference Optimization for Generative Real-World Super-Resolution},
  author={Fei, Song and Ye, Tian and Chen, Sixiang and Xing, Zhaohu and Lai, Jianyu and Zhu, Lei},
  journal={arXiv preprint arXiv:2603.05947},
  year={2026}
}
```

原项目：[W2GenAI-Lab/LucidNFT](https://github.com/W2GenAI-Lab/LucidNFT) — 觉得有用请给官方点星。
