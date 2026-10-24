# -*- coding: utf-8 -*-
"""把 LucidNFT 的 `lora_dit`（PEFT adapter）转成标准 comfy LoRA，供原生 LoraLoaderModelOnly 使用。

这样插件里就不需要自己的 LoRA 节点了：
    UNETLoader(flux-dev) -> LoraLoaderModelOnly(LucidNFT_lora_dit) -> LucidNFTSampler

只处理 DiT 这一支。条件分支的 `lora_condition` 目标权重不在主模型里，转不成 LoRA，
仍然由 LucidNFTCondition 的 condition_lora 下拉框在装载时合并（不落盘）。

依赖只有 torch + safetensors（用 ComfyUI 的 python 跑即可）：

    python tools/convert_dit_lora.py                  # 用默认路径转换
    python tools/convert_dit_lora.py --dry-run        # 只看会做什么
    python tools/convert_dit_lora.py --force          # 覆盖已存在文件
    python tools/convert_dit_lora.py --alpha 32        # 手动指定 alpha（默认读 adapter_config.json）
    python tools/convert_dit_lora.py --dtype bf16      # 输出精度（默认沿用原张量精度）
"""
import argparse
import json
import os
import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
DEFAULT_MODELS = _HERE.parents[2] / "models"      # custom_nodes/插件/tools -> ComfyUI
OUT_NAME = "LucidNFT_lora_dit.safetensors"

# loRA 的目标必须落在 flux DiT 的这些模块上，否则说明这份 adapter 不是 lora_dit
VALID_PREFIXES = ("diffusion_model.double_blocks.", "diffusion_model.single_blocks.")
PREFIXES = ("base_model.model.", "base_model.", "model.")


def load_adapter(path):
    """读 PEFT 目录（或单个权重文件）-> (state_dict, alpha, rank, weights_path)。"""
    path = Path(path)
    if path.is_dir():
        if not (path / "adapter_config.json").is_file():
            raise SystemExit(f"{path} 下没有 adapter_config.json，不像是 PEFT adapter 目录")
        for f in ("adapter_model.safetensors", "adapter_model.bin"):
            if (path / f).is_file():
                weights = path / f
                break
        else:
            raise SystemExit(f"{path} 下找不到 adapter_model.safetensors/.bin")
        cfg = json.loads((path / "adapter_config.json").read_text(encoding="utf-8"))
        alpha, rank = cfg.get("lora_alpha"), cfg.get("r")
    else:
        weights, alpha, rank = path, None, None

    if weights.suffix.lower() == ".safetensors":
        from safetensors.torch import load_file
        sd = load_file(str(weights))
    else:
        sd = torch.load(str(weights), map_location="cpu", weights_only=True)
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]

    if rank is None:
        for k, v in sd.items():
            if k.endswith(".lora_A.weight"):
                rank = int(v.shape[0])
                break
    if rank is None:
        raise SystemExit("文件里没有 *.lora_A.weight，不是 PEFT LoRA")
    if alpha is None:
        alpha = rank
        print(f"        注意：没有 adapter_config.json，alpha 按 rank={rank} 处理（scale=1.0）")
    return sd, float(alpha), int(rank), weights


def strip_prefix(key):
    for p in PREFIXES:
        if key.startswith(p):
            return key[len(p):]
    return key


def to_comfy(sd, alpha):
    """-> {diffusion_model.<module>.lora_A/B.weight, .alpha}"""
    out, modules = {}, {}
    for k, v in sd.items():
        for tag in ("A", "B"):
            suf = f".lora_{tag}.weight"
            if k.endswith(suf):
                modules.setdefault(strip_prefix(k[: -len(suf)]), {})[tag] = v
    for base in modules:
        ab = modules[base]
        if "A" not in ab or "B" not in ab:
            print(f"        !! {base} 缺少 lora_A 或 lora_B，跳过")
            continue
        key = "diffusion_model." + base
        out[key + ".lora_A.weight"] = ab["A"]
        out[key + ".lora_B.weight"] = ab["B"]
        out[key + ".alpha"] = torch.tensor(float(alpha), dtype=torch.float32)
    return out, sorted("diffusion_model." + b for b in modules)


def main(argv=None):
    p = argparse.ArgumentParser(description="LucidNFT lora_dit (PEFT) -> 标准 comfy LoRA")
    p.add_argument("--models", default=str(DEFAULT_MODELS), help="ComfyUI/models 目录")
    p.add_argument("--adapter", default=None, help="默认 <models>/LucidFlux/lora_dit")
    p.add_argument("--output", "-o", default=None, help=f"默认 <models>/loras/{OUT_NAME}")
    p.add_argument("--alpha", type=float, default=None, help="缺省读 adapter_config.json 的 lora_alpha")
    p.add_argument("--dtype", choices=["keep", "bf16", "fp16", "fp32"], default="keep")
    p.add_argument("--force", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)

    models = Path(args.models)
    adapter = Path(args.adapter) if args.adapter else models / "LucidFlux" / "lora_dit"
    out_path = Path(args.output) if args.output else models / "loras" / OUT_NAME
    if not adapter.exists():
        raise SystemExit(f"找不到 adapter: {adapter}")

    sd, alpha, rank, weights = load_adapter(adapter)
    alpha = float(args.alpha) if args.alpha is not None else alpha
    lora, modules = to_comfy(sd, alpha)

    bad = [m for m in modules if not m.startswith(VALID_PREFIXES)]
    print(f"输入: {weights}")
    print(f"模块: {len(modules)} 个（{len(modules) - len(bad)} 个落在 flux DiT 命名空间内）")
    print(f"alpha={alpha:g}  rank={rank}  ->  scale={alpha / rank:.4f}")
    if bad:
        print(f"!! 有 {len(bad)} 个模块不在 double_blocks/single_blocks 下，例: {bad[:3]}")
        print("   这份 adapter 可能不是 lora_dit，转换后 comfy 会加载不上")
        return 1

    dtype = {"keep": None, "bf16": torch.bfloat16, "fp16": torch.float16,
             "fp32": torch.float32}[args.dtype]
    if dtype is not None:
        lora = {k: (v.to(dtype) if v.is_floating_point() and v.dim() else v) for k, v in lora.items()}

    if args.dry_run:
        print(f"[dry-run] 将写出 {out_path}（{len(lora)} 张量）")
        return 0
    if out_path.exists() and not args.force:
        print(f"!! 已存在 {out_path}（加 --force 覆盖）")
        return 1

    from safetensors.torch import save_file
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(lora, str(out_path), metadata={
        "format": "pt",
        "target_format": "ComfyUI generic LoRA",
        "source_format": "PEFT LoRA (LucidNFT lora_dit)",
        "training_rank": str(rank),
        "training_alpha": str(alpha),
        "converted_by": "ComfyUI_LucidNFT/tools/convert_dit_lora.py",
    })
    print(f"写出: {out_path}  ({out_path.stat().st_size / 1024 / 1024:.2f} MB, {len(lora)} 张量)")
    print("用法: LoraLoaderModelOnly 选 "
          f"{out_path.name}，strength_model=1.0；插件里不用再接 LoRA 节点。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
