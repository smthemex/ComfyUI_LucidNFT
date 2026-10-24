# ComfyUI_LucidNFT

[LucidNFT](https://github.com/W2GenAI-Lab/LucidNFT)：LR-Anchored Multi-Reward Preference Optimization for Generative Real-World Super-Resolution

> **v2：改成和 [ComfyUI_LucidFlux](https://github.com/smthemex/ComfyUI_LucidFlux) 同一条 comfy 原生管线**
> —— 主 DiT 用 `UNETLoader` 直接吃，条件分支包成 comfy 的 `ControlBase`，
> 采样走 comfy 原生 euler + 官方 sigma 表，VAE / LoRA / 显存管理全部交给 comfy。
> 不再需要 `diffusers` / `transformers` / `peft` / `accelerate`，也没有 PiD 后段。
> 插件只有 **5 个节点**：DiT 侧的 LoRA 已转成标准 comfy LoRA，用原生 `LoraLoaderModelOnly` 加载。
>
> v1 的 diffusers wrapper（`src/`、`LucidNFT_node.py`、`inference.py`、`model_loader_utils.py`）
> 已删除，备份见 git 的 `snapshot: legacy diffusers implementation` 提交。

---

## 1. 安装

`custom_nodes/` 下：

```
git clone https://github.com/smthemex/ComfyUI_LucidNFT
```

`requirements.txt` 只剩 `torch` + `einops`（ComfyUI 自带）。没有别的第三方依赖，
不会动你环境里的 transformers / diffusers 版本。

## 2. 权重

```
ComfyUI/models/
├── diffusion_models/flux1-dev-fp8.safetensors        # 任意标准 flux DiT（推荐 fp8）
├── vae/ae.safetensors                               # 官方 flux ae（/8）
├── vae/UltraFlux.safetensors                        # 可选：出 2K 时用（F16 /16）
├── clip_vision/siglip2-so400m-patch16-512.safetensors  # 建议改个名，否则都叫 model.safetensors
└── LucidFlux/                                       # 与 ComfyUI_LucidFlux 共用同一目录
    ├── general_swinir_v1.ckpt                       # DiffBIR 先验
    ├── lucidflux.pth                                # 双条件分支（condition_lq/ldr + modulation）
    ├── prompt_embeddings.pt                         # 官方预计算文本 embedding（txt 512×4096 + vec 768）
    ├── lucid_connector.pth                          # SigLIP→4096 的 Redux 映射（也可直接给 lucidflux.pth）
    ├── lora_dit/{adapter_config.json, adapter_model.safetensors}        # LucidNFT 的 DiT LoRA（PEFT 原始格式）
    └── lora_condition/{adapter_config.json, adapter_model.safetensors}  # LucidNFT 的条件分支 LoRA（PEFT 原始格式）

转换后（跑一次 tools/convert_dit_lora.py 得到，见第 3 节）：
models/loras/LucidNFT_lora_dit.safetensors   # lora_dit 转成的标准 comfy LoRA
```

下载：`lucidflux.pth` / `prompt_embeddings.pt` → [W2GenAI/LucidFlux](https://huggingface.co/W2GenAI/LucidFlux/tree/main)；
`lora_dit` / `lora_condition` → [W2GenAI/LucidNFT](https://huggingface.co/W2GenAI/LucidNFT/tree/main)；
SigLIP → [google/siglip2-so400m-patch16-512](https://huggingface.co/google/siglip2-so400m-patch16-512/tree/main)（只下单体模型）；
SwinIR → [lxq007/DiffBIR](https://huggingface.co/lxq007/DiffBIR/tree/main) 的 `general_swinir_v1.ckpt`；
connector → [smthem/LucidFLUX-connector](https://huggingface.co/smthem/LucidFLUX-connector)。

> **文本不走 CLIP/T5**：直接读官方 `prompt_embeddings.pt`（与 CLIP 推理等价），
> 所以整条链**不要连 CLIPTextEncode / DualCLIPLoader**，省 5.4GB 权重和一次文本前向。

## 3. 五个节点 + 一条 LoRA 转换命令

两个 LoRA 的目标不一样，处理方式也不同，所以插件里**没有 LoRA 节点**：

| LoRA | 目标权重 | 处理 |
|---|---|---|
| `lora_dit` | 主 flux DiT 的 `double_blocks.*.img_attn / txt_attn` | 转成标准 comfy LoRA，用**原生 `LoraLoaderModelOnly`** 加载（本插件不参与） |
| `lora_condition` | 条件分支的 `double_blocks.*` + `modulation_*`（主模型里没有这些权重） | 由 `LucidNFTCondition` 的 `condition_lora` 下拉框在装载时合并，不落盘 |

一次性转换（用 ComfyUI 的 python 跑，只依赖 torch + safetensors）：

```
python tools/convert_dit_lora.py
# 输入 models/LucidFlux/lora_dit（PEFT adapter 目录）
# 输出 models/loras/LucidNFT_lora_dit.safetensors
# alpha 自动取 adapter_config.json 的 lora_alpha=64、rank=32 -> scale 2.0
```

转换是**逐张量无损**的：A/B 张量、alpha、76 个模块名与原来在插件里运行时生成的 patch 完全一致（实测 max|Δ|=0.0）。

| 节点 | 作用 |
|---|---|
| `LucidNFT DiffBIR (SwinIR)` | SwinIR 通用复原，输出 `ldr`（复原图）+ `lq`（缩放后的输入），两者都是 latent×8 |
| `LucidNFT Condition` | 文本 pt + SigLIP→Redux(1024 token) + 双条件分支（含 `lora_condition`）+ guidance，一条 CONDITIONING 输出 |
| `LucidNFT Sampler` | 官方 `get_schedule` 的精确 sigma 表 + comfy 原生 euler（cfg=1.0） |
| `LucidNFT VAE Patch (F16)` | 只有用 `UltraFlux` VAE 出 2K 时才接；用 flux ae 时不要接 |
| `LucidNFT Color Align` | wavelet 颜色对齐（reference 接 `ldr`） |

## 4. 工作流

`example_workflows/`

* `lucidnft_1k_comfy_core.json` — 11 节点 / 12 连线，latent 128 → **1024²**（flux ae）
* `lucidnft_2k_comfy_core.json` — 12 节点 / 13 连线，latent 128 → **2048²**（UltraFlux + F16 补丁）

```
UNETLoader(flux1-dev) → LoraLoaderModelOnly(LucidNFT_lora_dit) ────┐
LoadImage → LucidNFTDiffBIR(填 1024 = latent×8) ──lq/ldr──┐        │
                                                          ├→ LucidNFTCondition → LucidNFTSampler
EmptySD3LatentImage(1024 → latent 128) ───────────────────┘        │
VAELoader(ae) → VAEDecode → LucidNFTColorAlign(reference=ldr) → SaveImage
```

* `EmptySD3LatentImage` 填的是**像素**尺寸（内部 /8）：要 latent 128 就填 1024。
* `LucidNFTDiffBIR` 的 width/height 填 **latent×8**（latent 128 → 1024），也就是分支条件图的尺寸。
* 出 2K：VAE 换 `UltraFlux.safetensors` 并在 VAELoader 后接 `LucidNFTVAEPatch`，同一 latent 解出 2048²。

## 5. 参数

| 参数 | 建议 |
|---|---|
| steps | 官方 24（dev、无加速 lora）；接 Turbo-Alpha 可 4~8 |
| guidance | 4.0（在 `LucidNFTCondition` 里，不在 Sampler） |
| cfg | **恒 1.0**（flux-dev 是 guidance 蒸馏模型，不要 CFG） |
| shift | flux-dev 开；flux-schnell 关 |
| dtype | bf16（12GB 卡也可跑） |
| lora_strength | 1.0；想更“贴原图”可降到 0.6~0.8 |

实测（RTX 4070 12GB，flux1-dev-fp8 + bf16 分支 + 两个 LoRA，1024²）：
20 步 **52s**（≈2.4s/step），峰值显存 **8.2GB**；VAE 解码 0.6s。
同 seed 下开/关 LoRA 的输出差异 mean|Δ|≈4.2/255（LoRA 生效，画质细节取向变化）。

## 6. 原生链顺手解决的问题

| 老问题（v1） | 现在 |
|---|---|
| 接标准 VAE 节点有**色差**（v1 手动 `x/0.3611+0.1159` 再 decode） | latent 语义交给 comfy，`VAEDecode` 直接出正确颜色；`LucidNFTColorAlign` 只做 wavelet 高频/低频融合 |
| 自带采样循环，vram / offload 自己管 | 官方 sigma 表 + `comfy.sample.sample_custom`，显存交给 comfy（`ControlBase.get_models()` 注册 patcher） |
| 8 个节点、接线容易顺序错（LoRA 与 conditioning 顺序、SigLIP 连接器分开） | 5 个节点，pt/分支/connector/SigLIP 都在 Condition 里，只连 lq/ldr 两根线；LoRA 直接用 comfy 原生 `LoraLoaderModelOnly` |
| 依赖 diffusers/transformers/peft，transformers 升级就整插件消失 | 只依赖 torch + einops |

## 7. Citation

```
@article{fei2026lucidnft,
  title={LucidNFT: LR-Anchored Multi-Reward Preference Optimization for Generative Real-World Super-Resolution},
  author={Fei, Song and Ye, Tian and Chen, Sixiang and Xing, Zhaohu and Lai, Jianyu and Zhu, Lei},
  journal={arXiv preprint arXiv:2603.05947},
  year={2026}
}
```

原项目：[W2GenAI-Lab/LucidNFT](https://github.com/W2GenAI-Lab/LucidNFT) — 觉得有用请给官方点星。
