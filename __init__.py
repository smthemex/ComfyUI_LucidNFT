"""ComfyUI_LucidNFT — ComfyUI 原生 DiT 推理栈（不再依赖 diffusers / transformers / peft）。

LucidNFT = LucidFlux（双条件分支 ControlNet 型 DiT）+ 偏好优化 LoRA。

链路只用到 comfy 原生能力 + torch：

    UNETLoader(flux-dev) → LucidNFTLoRA(lora_dit) ─────────────────┐
    LoadImage → LucidNFTDiffBIR → LucidNFTCondition ───────────────┼→ LucidNFTSampler
    EmptySD3LatentImage ──────────────────────────────────────────┘
    VAELoader(flux ae) → VAEDecode → LucidNFTColorAlign → SaveImage
    （出 2K 时 VAELoader(UltraFlux) → LucidNFTVAEPatch → VAEDecode）

没有 PiD 4K 后段。旧的 diffusers wrapper（src/、inference.py、LucidNFT_node.py、
model_loader_utils.py、config.json）已移除，备份见
workbuddy/_LucidNFT_legacy_diffusers_backup.zip 或 git 的 snapshot 提交。
"""
import os

import folder_paths

_MODELS_DIR = os.path.join(folder_paths.models_dir, "LucidFlux")
if not os.path.isdir(_MODELS_DIR):
    os.makedirs(_MODELS_DIR, exist_ok=True)
# 与 ComfyUI_LucidFlux 共用同一个模型目录；重复注册会让下拉框出现重复项
if "LucidFlux" not in folder_paths.folder_names_and_paths:
    folder_paths.add_model_folder_path("LucidFlux", _MODELS_DIR)

from .lucidnft_nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS  # noqa: E402

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
