"""LucidNFT -> ComfyUI 原生 DiT 推理栈的核心层。

上游：https://github.com/W2GenAI-Lab/LucidNFT
      LucidNFT = LucidFlux（双条件分支 ControlNet 型 DiT）+ 偏好优化 LoRA。

结构（逐字核对 checkpoint 后的结论，与 ComfyUI_LucidFlux 完全同构）：
    flux-dev DiT（与 stock flux 完全同构，用 comfy 原生 UNETLoader 直接吃）
  + dual condition branch：两个 ControlNet 式分支（LQ / LDR），各 743.8M
      每个分支 = img_in/time_in/vector_in/guidance_in/txt_in
               + 2 个 DoubleStreamBlock（controlnet_depth=2）
               + 2 个 controlnet_blocks（3072→3072）
               + input_hint_block（3 次 stride=2 卷积 = /8）
               + pos_embed_input
  + modulation_lq / modulation_ldr：按 control_index(0..18) 对分支输出做
    逐 block 的 (shift, scale) 调制（time embed + control-index embed）
  + LoRA：`lora_dit`（DiT 的 img_attn/txt_attn，走 comfy 原生 LoRA patch）
          `lora_condition`（两条分支的 attn + modulation.linear，装载时合并进权重）

每个去噪步：residual[k] = mod_lq(out_lq[k%2], t, k) + mod_ldr(out_ldr[k%2], t, k)，
在 double block k 之后以 `img = img + residual[k]` 注入 img 流。

comfy 原生 flux 的 control 注入点（comfy/ldm/flux/model.py）就是
`img[:, :add.shape[1]] += add`，语义逐位一致——因此本插件只需要：
  1. 用 comfy 原生模块（comfy.ldm.flux.layers）拼出分支，key 与 checkpoint 1:1；
  2. 把 19 个 residual 包成 {"input": [...], "output": []} 交给采样器。
主模型加载 / 采样 / VAE / 显存管理全部走 comfy 原生。

与 ComfyUI_LucidFlux 的差异（本插件只做 LucidNFT 需要的那部分）：
  * 增加 LoRA 支持（DiT + 条件分支），这是 LucidNFT 的本体；
  * 没有 PiD 后段；
  * 默认 VAE 是官方 flux ae（/8），F16(/16) 补丁只在用 UltraFlux VAE 出 2K 时才接。
"""
import json
import logging
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

import comfy.controlnet
import comfy.ldm.flux.layers as flux_layers
import comfy.model_management
import comfy.model_patcher
import comfy.ops
import comfy.sample
import comfy.samplers
import comfy.sd
import comfy.utils

ops = comfy.ops.manual_cast
logger = logging.getLogger(__name__)

HIDDEN = 3072
NUM_HEADS = 24
AXES_DIM = (16, 56, 56)
THETA = 10000
IN_CHANNELS = 64
CONTEXT_DIM = 4096
VEC_DIM = 768
CTRL_DEPTH = 2          # controlnet_depth：每个分支的 double block 数
MAIN_DEPTH = 19         # 主 DiT 的 double block 数（= residual 个数）
PATCH = 2

STATE_KEYS = ("condition_lq", "condition_ldr", "modulation_lq", "modulation_ldr")

# PEFT adapter 里可能出现的包装前缀（官方训练脚本是 base_model.model.）
_ADAPTER_PREFIXES = ("base_model.model.", "base_model.", "model.")


# ---------------------------------------------------------------------------
# 分支模块：key 与 checkpoint 1:1（load_state_dict(strict=True) 自校验）
# ---------------------------------------------------------------------------

class _TimeEmbedding(nn.Module):
    """key 名对齐 checkpoint：linear_1 / linear_2（SiLU 无参数）。"""

    def __init__(self, in_dim=256, out_dim=HIDDEN, dtype=None, device=None):
        super().__init__()
        self.linear_1 = ops.Linear(in_dim, out_dim, bias=True, dtype=dtype, device=device)
        self.act = nn.SiLU()
        self.linear_2 = ops.Linear(out_dim, out_dim, bias=True, dtype=dtype, device=device)

    def forward(self, x):
        return self.linear_2(self.act(self.linear_1(x)))


class LucidNFTModulation(nn.Module):
    """官方 Modulation：norm(x) * (1 + scale) + shift，scale/shift 由 t 与 control_index 决定。"""

    def __init__(self, dim=HIDDEN, dtype=None, device=None):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = ops.Linear(dim, 2 * dim, bias=True, dtype=dtype, device=device)
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.timestep_embedder = _TimeEmbedding(256, dim, dtype=dtype, device=device)
        self.control_index_embedder = _TimeEmbedding(256, dim, dtype=dtype, device=device)

    def forward(self, x, sigma, control_index):
        t_emb = flux_layers.timestep_embedding(sigma, 256).to(dtype=x.dtype)      # sigma*1000
        idx = torch.full_like(sigma, float(control_index))
        # 官方：control_index 不做 *1000（Timesteps.scale=1）
        idx_emb = flux_layers.timestep_embedding(idx, 256, time_factor=1.0).to(dtype=x.dtype)
        emb = self.linear(self.silu(self.timestep_embedder(t_emb) + self.control_index_embedder(idx_emb)))
        shift, scale = emb.chunk(2, dim=-1)
        return self.norm(x) * (1 + scale[:, None]) + shift[:, None]


def _hint_block(dtype=None, device=None):
    conv = lambda cin, cout, **kw: ops.Conv2d(cin, cout, 3, padding=1, dtype=dtype, device=device, **kw)
    return nn.Sequential(
        conv(3, 16), nn.SiLU(),
        conv(16, 16), nn.SiLU(),
        conv(16, 16, stride=2), nn.SiLU(),
        conv(16, 16), nn.SiLU(),
        conv(16, 16, stride=2), nn.SiLU(),
        conv(16, 16), nn.SiLU(),
        conv(16, 16, stride=2), nn.SiLU(),
        conv(16, 16),
    )


class LucidNFTControlBranch(nn.Module):
    """官方 SingleConditionBranch（controlnet_depth=2）。"""

    def __init__(self, dtype=None, device=None):
        super().__init__()
        self.hidden_size = HIDDEN
        self.num_heads = NUM_HEADS
        self.pe_embedder = flux_layers.EmbedND(dim=HIDDEN // NUM_HEADS, theta=THETA, axes_dim=list(AXES_DIM))
        self.img_in = ops.Linear(IN_CHANNELS, HIDDEN, bias=True, dtype=dtype, device=device)
        self.time_in = flux_layers.MLPEmbedder(256, HIDDEN, dtype=dtype, device=device, operations=ops)
        self.vector_in = flux_layers.MLPEmbedder(VEC_DIM, HIDDEN, dtype=dtype, device=device, operations=ops)
        self.guidance_in = flux_layers.MLPEmbedder(256, HIDDEN, dtype=dtype, device=device, operations=ops)
        self.txt_in = ops.Linear(CONTEXT_DIM, HIDDEN, bias=True, dtype=dtype, device=device)
        self.double_blocks = nn.ModuleList([
            flux_layers.DoubleStreamBlock(HIDDEN, NUM_HEADS, mlp_ratio=4.0, qkv_bias=True,
                                          dtype=dtype, device=device, operations=ops)
            for _ in range(CTRL_DEPTH)
        ])
        self.controlnet_blocks = nn.ModuleList([
            ops.Linear(HIDDEN, HIDDEN, bias=True, dtype=dtype, device=device) for _ in range(CTRL_DEPTH)
        ])
        self.pos_embed_input = ops.Linear(IN_CHANNELS, HIDDEN, bias=True, dtype=dtype, device=device)
        self.input_hint_block = _hint_block(dtype=dtype, device=device)

    def forward(self, img, img_ids, cond_img, txt, txt_ids, y, sigma, guidance):
        img = self.img_in(img)

        hint = self.input_hint_block(cond_img)
        hint = rearrange(hint, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=PATCH, pw=PATCH)
        img = img + self.pos_embed_input(hint)

        vec = self.time_in(flux_layers.timestep_embedding(sigma, 256).to(img.dtype))
        vec = vec + self.guidance_in(flux_layers.timestep_embedding(guidance, 256).to(img.dtype))
        vec = vec + self.vector_in(y)

        txt = self.txt_in(txt)
        pe = self.pe_embedder(torch.cat((txt_ids, img_ids), dim=1))

        res = []
        # 注意：comfy 的 DoubleStreamBlock 是**原地** `img += ...`，
        # 若先收集 img 再统一投影，前面收集的引用会被后续 block 改写（别名 bug）。
        # 所以在每个 block 之后立刻做 controlnet 投影（投影产生新张量）。
        for cb, block in zip(self.controlnet_blocks, self.double_blocks):
            img, txt = block(img=img, txt=txt, vec=vec, pe=pe, transformer_options={})
            res.append(cb(img))
        return res


class LucidNFTControlModel(nn.Module):
    """官方 DualConditionBranch + 两个 Modulation 的等价实现。"""

    def __init__(self, dtype=None, device=None):
        super().__init__()
        self.lq = LucidNFTControlBranch(dtype=dtype, device=device)
        self.ldr = LucidNFTControlBranch(dtype=dtype, device=device)
        self.modulation_lq = LucidNFTModulation(dtype=dtype, device=device)
        self.modulation_ldr = LucidNFTModulation(dtype=dtype, device=device)

    def forward(self, img, img_ids, cond_lq, cond_ldr, txt, txt_ids, vec, sigma, guidance):
        out_lq = self.lq(img, img_ids, cond_lq, txt, txt_ids, vec, sigma, guidance)
        out_ldr = self.ldr(img, img_ids, cond_ldr, txt, txt_ids, vec, sigma, guidance)

        out = []
        for k in range(MAIN_DEPTH):
            j = k % CTRL_DEPTH
            out.append(self.modulation_lq(out_lq[j], sigma, k) + self.modulation_ldr(out_ldr[j], sigma, k))
        return out


# ---------------------------------------------------------------------------
# PEFT LoRA（adapter_config.json + adapter_model.safetensors）
# ---------------------------------------------------------------------------

_ADAPTER_WEIGHT_FILES = ("adapter_model.safetensors", "adapter_model.bin")


def resolve_adapter_dir(path):
    """给定目录或其内文件，返回 (adapter_dir, weights_path)；非 adapter 结构返回 (None, path)。"""
    if path is None:
        return None, None
    if os.path.isdir(path):
        if not os.path.isfile(os.path.join(path, "adapter_config.json")):
            return None, None
        for f in _ADAPTER_WEIGHT_FILES:
            p = os.path.join(path, f)
            if os.path.isfile(p):
                return path, p
        return None, None
    return None, path


def read_peft_adapter(path, verbose=True):
    """读 PEFT adapter，返回 (state_dict, alpha, rank)。

    alpha 优先取 adapter_config.json 的 lora_alpha（官方是 64，r=32 → scale 2.0）；
    没有 config 时退回 alpha=rank（即 scale=1.0）并告警。
    """
    adapter_dir, weights = resolve_adapter_dir(path)
    if weights is None:
        raise ValueError(f"找不到 LoRA 权重: {path}")

    if weights.endswith(".safetensors"):
        from safetensors.torch import load_file
        sd = load_file(weights)
    else:
        sd = torch.load(weights, map_location="cpu", weights_only=True)
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]

    alpha, rank = None, None
    cfg_path = os.path.join(adapter_dir, "adapter_config.json") if adapter_dir else None
    if cfg_path and os.path.isfile(cfg_path):
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        alpha = cfg.get("lora_alpha", None)
        rank = cfg.get("r", None)

    for k, v in sd.items():
        if k.endswith(".lora_A.weight"):
            rank = rank or int(v.shape[0])
            break
    if rank is None:
        raise ValueError(f"{path} 里没有 lora_A/lora_B 键，不像是 PEFT adapter")
    if alpha is None:
        alpha = rank
        if verbose:
            logger.warning("[LucidNFT] %s 缺少 adapter_config.json，alpha 按 rank 处理（scale=1.0）", path)
    return sd, float(alpha), int(rank)


def _strip_adapter_prefix(key):
    for p in _ADAPTER_PREFIXES:
        if key.startswith(p):
            return key[len(p):]
    return key


def _map_condition_key(key):
    """adapter 里的模块路径 -> 本插件 LucidNFTControlModel 的模块路径。"""
    key = _strip_adapter_prefix(key)
    if key.startswith("controlnet."):
        key = key[len("controlnet."):]
    # 官方 DualConditionComposer：lq / pre(=ldr) / modulation_lq / modulation_pre(=modulation_ldr)
    if key.startswith("modulation_pre."):
        return "modulation_ldr." + key[len("modulation_pre."):]
    if key.startswith("pre."):
        return "ldr." + key[len("pre."):]
    return key


def merge_condition_lora(model, adapter_path, strength=1.0, verbose=True):
    """把 lora_condition 合并进条件分支权重（原地换 Parameter，不碰 mmap 源张量）。

    PEFT: delta_W = B @ A * (alpha / rank)；本函数额外乘 strength。
    """
    sd, alpha, rank = read_peft_adapter(adapter_path, verbose=verbose)
    scale = (alpha / float(rank)) * float(strength)

    modules = sorted({k.rsplit(".lora_A.weight", 1)[0] for k in sd if k.endswith(".lora_A.weight")})
    params = dict(model.named_parameters())
    merged, skipped = 0, []
    for m in modules:
        a = sd.get(f"{m}.lora_A.weight")
        b = sd.get(f"{m}.lora_B.weight")
        if a is None or b is None:
            skipped.append(m)
            continue
        target = _map_condition_key(m) + ".weight"
        p = params.get(target)
        if p is None:
            skipped.append(f"{m} -> {target}")
            continue
        if p.shape != (b.shape[0], a.shape[1]):
            skipped.append(f"{m} shape {tuple(b.shape)}@{tuple(a.shape)} vs {tuple(p.shape)}")
            continue
        delta = (b.float() @ a.float()) * scale
        # checkpoint 里的分支权重是 bf16：先在 fp32 里累加、保持 fp32，
        # 由最后的 model.to(dtype) 做一次性舍入（比「先舍到 bf16 再加」少一次精度损失）。
        new = p.detach().float() + delta.to(p.device)
        p.data = new.contiguous()
        merged += 1

    if verbose:
        logger.info("[LucidNFT] condition LoRA merged: %d/%d modules (scale=%.3f)", merged, len(modules), scale)
        if skipped:
            logger.warning("[LucidNFT] condition LoRA 未匹配: %s", skipped[:6])
    if merged == 0:
        raise ValueError(f"LoRA 与条件分支不匹配（0 个模块被合并）: {adapter_path}")
    del sd
    return merged


def build_dit_lora_dict(adapter_path, verbose=True):
    """把 PEFT adapter 的 key 改写成 comfy 通用 LoRA 格式（diffusion_model.*）。

    comfy 的 LoRAAdapter 直接认 `.lora_A.weight` / `.lora_B.weight`（A=down, B=up），
    只需要把包装前缀换成 `diffusion_model.`，并补 `{module}.alpha`（否则 alpha 缺失）。
    """
    sd, alpha, rank = read_peft_adapter(adapter_path, verbose=verbose)
    lora, modules = {}, set()
    for k, v in sd.items():
        if k.endswith(".lora_A.weight") or k.endswith(".lora_B.weight"):
            nk = "diffusion_model." + _strip_adapter_prefix(k)
            lora[nk] = v
            modules.add(nk.rsplit(".lora_", 1)[0])
    for m in modules:
        lora[m + ".alpha"] = torch.tensor(float(alpha))
    if verbose:
        logger.info("[LucidNFT] DiT LoRA: %d modules, alpha=%.1f rank=%d", len(modules), alpha, rank)
    return lora, alpha, rank


def apply_dit_lora(model, adapter_path, strength=1.0, verbose=True):
    """给 comfy 的 MODEL 打 DiT LoRA（走 comfy 原生 patch，fp8/低显存都安全）。"""
    lora, _, _ = build_dit_lora_dict(adapter_path, verbose=verbose)
    new_model, _ = comfy.sd.load_lora_for_models(model, None, lora, strength, 0.0)
    if verbose:
        patched = len(getattr(new_model, "patches", {}) or {})
        logger.info("[LucidNFT] DiT LoRA applied (strength=%.2f), patched tensors=%d", strength, patched)
    return new_model


def apply_plain_lora(model, lora_path, strength=1.0):
    """给 comfy 的 MODEL 打标准 comfy 格式 LoRA（models/loras 里的 turbo lora 等）。"""
    sd = comfy.utils.load_torch_file(lora_path, safe_load=True)
    new_model, _ = comfy.sd.load_lora_for_models(model, None, sd, strength, 0.0)
    return new_model


# ---------------------------------------------------------------------------
# 装载
# ---------------------------------------------------------------------------

def load_control_model(path, dtype=torch.bfloat16, condition_lora=None, lora_strength=1.0, verbose=True):
    """读取 lucidflux.pth 的 4 个子 dict，严格装载到 comfy 原生模块，再按需合并 lora_condition。

    官方 flux 的 QKNorm 参数名是 `*.norm.query_norm.scale`，而 comfy 的
    `operations.RMSNorm`（torch.nn.RMSNorm）叫 `.weight` —— 与 comfy 自己加载
    flux1-dev 时的 `supported_models` 改名规则一致，这里做同样的 .scale -> .weight。
    """
    sd = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    missing = [k for k in STATE_KEYS if k not in sd]
    if missing:
        raise ValueError(f"{path} 缺少 LucidNFT/LucidFlux 权重段: {missing}（文件内只有 {list(sd.keys())}）")

    model = LucidNFTControlModel(dtype=dtype)
    report = {}
    for name, module in (("condition_lq", model.lq), ("condition_ldr", model.ldr),
                         ("modulation_lq", model.modulation_lq), ("modulation_ldr", model.modulation_ldr)):
        state = {k[:-6] + ".weight" if k.endswith(".scale") else k: v for k, v in sd[name].items()}
        # assign=True：直接接管 mmap 张量，省一份 3GB 峰值拷贝
        result = module.load_state_dict(state, strict=True, assign=True)
        report[name] = (len(result.missing_keys), len(result.unexpected_keys))
    del sd

    for p in model.parameters():
        p.requires_grad_(False)

    if condition_lora:
        merge_condition_lora(model, condition_lora, strength=lora_strength, verbose=verbose)

    model.eval()
    if dtype is not None:
        model = model.to(dtype)
    if verbose:
        n = sum(p.numel() for p in model.parameters())
        logger.info("[LucidNFT] control model loaded: %.1fM params, dtype=%s, strict-ok=%s",
                    n / 1e6, dtype, report)
    return model


def split_connector(src_path, verbose=True):
    """lucid_connector.pth，或直接给含 `connector` 段的 lucidflux.pth。"""
    sd = comfy.utils.load_torch_file(src_path, safe_load=False)
    if "redux_down.weight" in sd:
        return sd
    if "connector" in sd and isinstance(sd["connector"], dict):
        if verbose:
            logger.info("[LucidNFT] connector 取自 %s 的 'connector' 段", os.path.basename(src_path))
        return sd["connector"]
    raise ValueError(f"{src_path} 里既没有 redux_down.weight 也没有 connector 段")


# ---------------------------------------------------------------------------
# Control 对象（comfy ControlBase 子类，交给原生 sampler 调 get_control）
# ---------------------------------------------------------------------------

class LucidNFTControl(comfy.controlnet.ControlBase):
    """把 LucidNFT 的双分支包成 comfy 的 control 对象。

    原生采样器每步调用 get_control()，comfy flux 把返回的
    {"input": [19 x (B, N_img, 3072)]} 逐 block 加进 img 流。
    """

    def __init__(self, control_model, load_device=None):
        super().__init__()
        self.control_model = control_model
        self.load_device = load_device or comfy.model_management.get_torch_device()
        self.patcher = comfy.model_patcher.ModelPatcher(
            control_model,
            load_device=self.load_device,
            offload_device=comfy.model_management.unet_offload_device(),
        )
        self.cond_lq = None
        self.cond_ldr = None
        self.txt = None
        self.offload_device = comfy.model_management.unet_offload_device()

    # 复制体（共享权重与 patcher，只换条件图），语义同 ControlNetApply 的 copy()
    def copy_with(self, cond_lq, cond_ldr, txt):
        c = LucidNFTControl(self.control_model, self.load_device)
        c.patcher = self.patcher
        c.cond_lq = cond_lq
        c.cond_ldr = cond_ldr
        c.txt = txt
        return c

    def get_models(self):
        out = super().get_models()
        out.append(self.patcher)
        return out

    def inference_memory_requirements(self, dtype):
        return comfy.model_management.module_size(self.control_model) + super().inference_memory_requirements(dtype)

    def cleanup(self):
        # 只清 ControlBase 的临时状态；条件图必须保留 —— comfy 在采样结束后会调用
        # cleanup()，若这里把 cond_lq/txt 置空，同一份 conditioning 第二次采样（换 seed）
        # 会静默丢掉 control 引导。
        super().cleanup()

    # ---- 核心：每步算 residual -------------------------------------------
    def get_control(self, x_noisy, t, cond, batched_number, transformer_options):
        if self.cond_lq is None or self.cond_ldr is None or self.txt is None:
            return None

        sigma_t = transformer_options.get("sigmas", None)
        if sigma_t is None:
            sigma_value = float(t.detach().float().flatten()[0])
        else:
            sigma_value = float(sigma_t.detach().float().flatten()[0])

        comfy.model_management.load_models_gpu(
            [self.patcher], memory_required=self.inference_memory_requirements(torch.bfloat16))

        device = x_noisy.device
        dtype = self._model_dtype()
        b, c, h, w = x_noisy.shape

        img = rearrange(x_noisy.to(dtype), "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=PATCH, pw=PATCH)
        h_lat, w_lat = h // PATCH, w // PATCH
        img_ids = torch.zeros(h_lat, w_lat, 3, device=device, dtype=torch.float32)
        img_ids[..., 1] += torch.arange(h_lat, device=device, dtype=torch.float32)
        img_ids[..., 2] += torch.arange(w_lat, device=device, dtype=torch.float32)
        img_ids = repeat(img_ids, "h w c -> b (h w) c", b=b).to(dtype=dtype)

        txt = self.txt.to(device=device, dtype=dtype)
        txt_ids = torch.zeros(b, txt.shape[1], 3, device=device, dtype=dtype)

        vec = cond.get("pooled_output", None)
        if vec is None:
            vec = torch.zeros(b, VEC_DIM, device=device, dtype=dtype)
        else:
            vec = vec.to(device=device, dtype=dtype)
            if vec.shape[0] != b:
                vec = comfy.utils.repeat_to_batch_size(vec, b)

        g = cond.get("guidance", None)
        gv = float(g.detach().float().flatten()[0]) if g is not None else 4.0
        sigma = torch.full((b,), sigma_value, device=device, dtype=dtype)
        guidance = torch.full((b,), gv, device=device, dtype=dtype)

        # 分支条件图语义：像素边长 = latent * 8（官方 cond 图正是 latent*8）
        lq = self._hint(self.cond_lq, b, h, w, device, dtype)
        ldr = self._hint(self.cond_ldr, b, h, w, device, dtype)

        with torch.no_grad():
            out = self.control_model(img, img_ids, lq, ldr, txt, txt_ids, vec, sigma, guidance)
        return {"input": [x.to(dtype) for x in out], "output": []}

    def _model_dtype(self):
        try:
            return self.patcher.model_dtype()
        except Exception:
            return torch.bfloat16

    @staticmethod
    def _hint(img, batch, h_lat, w_lat, device, dtype):
        img = img.to(device=device, dtype=torch.float32)
        target_h, target_w = h_lat * 8, w_lat * 8
        if img.shape[-2] != target_h or img.shape[-1] != target_w:
            if img.shape[0] == 1 and batch > 1:
                img = img.repeat(batch, 1, 1, 1)
            img = comfy.utils.common_upscale(img, target_w, target_h, "bilinear", "disabled")
        if img.shape[0] != batch:
            img = comfy.utils.repeat_to_batch_size(img, batch)
        return img.to(dtype)


# ---------------------------------------------------------------------------
# 采样调度：官方 get_schedule 的逐字复刻
# ---------------------------------------------------------------------------

def _lin_function(x1=256.0, y1=0.5, x2=4096.0, y2=1.15):
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return lambda x: m * x + b


def _time_shift(mu, sigma, t):
    return math.exp(mu) / (math.exp(mu) + (1.0 / t - 1.0) ** sigma)


def official_image_seq_len(latent_h, latent_w):
    """官方 image_seq_len 的等价式。

    官方：(w//8)*(h//8)//(16*16)，其中 w = latent*8 → 等价 latent_h*latent_w/256
    （LucidFlux 的 2K 写法 (w//8)*(h//8)//(32*32) 配 w=latent*16 也是同式，
     所以 comfy 侧只看 latent 尺寸即可，不需要区分出图分辨率。）
    """
    return max(1, int(latent_h) * int(latent_w) // 256)


def build_sigmas(num_steps, latent_h, latent_w, shift=True):
    """返回 comfy euler 用的 sigma 表（= 官方 timesteps，含末尾 0）。"""
    seq = official_image_seq_len(latent_h, latent_w)
    ts = torch.linspace(1.0, 0.0, num_steps + 1, dtype=torch.float32)
    if shift:
        mu = _lin_function()(seq)
        exp_mu = math.exp(mu)
        ts = exp_mu / (exp_mu + (1.0 / ts - 1.0))
        ts = torch.nan_to_num(ts, nan=0.0, posinf=0.0, neginf=0.0)
    return ts.contiguous()


def sample_lucidnft(model, cond, latent_image, sigmas, noise=None, seed=0,
                    cfg=1.0, disable_pbar=False):
    sampler = comfy.samplers.sampler_object("euler")
    return comfy.sample.sample_custom(
        model, noise, cfg=cfg, sampler=sampler, sigmas=sigmas,
        positive=cond, negative=[], latent_image=latent_image,
        disable_pbar=disable_pbar, seed=seed)


# ---------------------------------------------------------------------------
# VAE：可选 F16(/16) 语义（只在使用 UltraFlux VAE 出 2K 时需要）
# ---------------------------------------------------------------------------

class _F16ConvOut(nn.Module):
    """官方 Decoder 尾部 conv_act -> nearest x2 -> conv_out。"""

    def __init__(self, conv, stride=2):
        super().__init__()
        self.conv = conv
        self.stride = stride

    def forward(self, x):
        if self.stride <= 1:
            return self.conv(x)
        return self.conv(F.interpolate(x, scale_factor=self.stride, mode="nearest"))


def patch_vae_f16(vae, stride=2):
    fs = getattr(vae, "first_stage_model", None)
    if fs is None or not hasattr(fs, "decoder"):
        raise ValueError("这不是 comfy 的 AutoencodingEngine VAE，无法打 F16 补丁")
    dec = fs.decoder
    if getattr(dec, "_lucidnft_f16", False):
        return vae
    try:
        fs.encoder.conv_in.stride = (stride, stride)
    except Exception:
        pass
    dec.conv_out = _F16ConvOut(dec.conv_out, stride=stride)
    dec._lucidnft_f16 = True
    try:
        vae.downscale_ratio = 8 * stride
        vae.upscale_ratio = 8 * stride
    except Exception:
        pass
    return vae
