"""LucidNFT 节点（ComfyUI 原生 DiT 链，6 个节点，无 PiD 后段）。

工作流（1K 复现）：

    LoadImage → LucidNFTDiffBIR ──(lq / ldr)──┐
                                               ├→ LucidNFTCondition → LucidNFTSampler
    （内部：prompt_embeddings.pt + lucidflux.pth（+lora_condition）+ Connector + SigLIP）
                                                                     │
    EmptySD3LatentImage ─────────────────────────────────────────────┤
    VAELoader(flux ae) → VAEDecode → LucidNFTColorAlign → SaveImage

    UNETLoader(flux-dev) → LucidNFTLoRA（lora_dit）→ 上面的 model

出 2K（官方 UltraFlux F16 VAE）时在 VAELoader 后面多接一个 LucidNFTVAEPatch。

与 ComfyUI_LucidFlux 的关系：同一条原生管线；本插件多了 LoRA（LucidNFT 的本体），
少了 PiD 4K 解码段。文本同样直接吃官方 prompt_embeddings.pt，不加载 CLIP/T5。
"""
import logging
import os

import torch

import comfy.clip_vision
import comfy.model_management as mm
import comfy.sample
import comfy.sd
import comfy.utils
import folder_paths
import node_helpers

from . import lucidnft_core as core
from .align_color import wavelet_reconstruction
from .swinir_model import load_swinir

MODELS_SUBDIR = "LucidFlux"
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 下拉框候选
# ---------------------------------------------------------------------------

def _list(substr=None):
    files = folder_paths.get_filename_list(MODELS_SUBDIR)
    if substr:
        files = [f for f in files if substr in f.lower()]
    return files or ["none"]


def _siglip_list():
    files = [f for f in folder_paths.get_filename_list("clip_vision") if "siglip" in f.lower()]
    return files or ["none"]


_ADAPTER_CACHE = {"scan": None, "dit": [], "condition": []}


def _scan_adapters():
    """扫 models/LucidFlux/*/adapter_model.* 并按用途分类（dit / condition）。

    LucidNFT 官方权重是 PEFT adapter 目录（adapter_config.json + adapter_model.safetensors）,
    不能直接当 comfy lora 文件用，所以这里按目录名列出。
    """
    bases = [os.path.join(folder_paths.models_dir, MODELS_SUBDIR)]
    try:
        bases += list(folder_paths.get_folder_paths(MODELS_SUBDIR))
    except Exception:
        pass
    bases = sorted({os.path.abspath(b) for b in bases if b and os.path.isdir(b)})

    stamp = []
    found = []
    for base in bases:
        for name in sorted(os.listdir(base)):
            d = os.path.join(base, name)
            if not os.path.isdir(d) or not os.path.isfile(os.path.join(d, "adapter_config.json")):
                continue
            w = None
            for f in ("adapter_model.safetensors", "adapter_model.bin"):
                if os.path.isfile(os.path.join(d, f)):
                    w = os.path.join(d, f)
                    break
            if w is None:
                continue
            found.append((name, w))
            stamp.append(w)

    if _ADAPTER_CACHE["scan"] == stamp:
        return

    dit, cond = [], []
    for name, w in found:
        try:
            if w.endswith(".safetensors"):
                from safetensors import safe_open
                with safe_open(w, framework="pt") as f:
                    keys = list(f.keys())
            else:
                keys = list(torch.load(w, map_location="meta", weights_only=True).keys())
        except Exception as e:  # 读不出来就两边都列上，让用户自己选
            logger.warning("[LucidNFT] 无法读取 %s 的键: %s", name, e)
            keys = []
        if any("controlnet." in k for k in keys) or not keys:
            cond.append(name)
        if any("double_blocks." in k and "controlnet." not in k for k in keys) or not keys:
            dit.append(name)

    _ADAPTER_CACHE.update({"scan": stamp, "dit": sorted(set(dit)), "condition": sorted(set(cond))})


def _adapter_list(kind):
    _scan_adapters()
    items = _ADAPTER_CACHE[kind]
    # 真实存在时排在前面（作为默认值），none 放最后保持向后兼容
    return items + ["none"]


def _resolve_adapter(name):
    if name in (None, "", "none"):
        return None
    for base in sorted({os.path.join(folder_paths.models_dir, MODELS_SUBDIR),
                        *folder_paths.get_folder_paths(MODELS_SUBDIR)}):
        d = os.path.join(base, name)
        if os.path.isfile(os.path.join(d, "adapter_config.json")):
            return d
    return None


# ---------------------------------------------------------------------------
# 图像小工具
# ---------------------------------------------------------------------------

def _image_to_bchw(image):
    return image.movedim(-1, 1).float()


def _bchw_to_image(x):
    return x.movedim(1, -1).float().clamp(0.0, 1.0)


def _resize_bchw(x, width, height):
    if width <= 0 or height <= 0:
        return x
    if x.shape[-1] == width and x.shape[-2] == height:
        return x
    return comfy.utils.common_upscale(x, width, height, "nearest-exact", "disabled")


# ---------------------------------------------------------------------------
# 重量级对象缓存（避免每次执行重读 3.4GB pth / connector / SigLIP）
# ---------------------------------------------------------------------------

_CACHE = {"control": {}, "style": {}, "clip_vision": {}}


def _get_control(control_path, dtype, condition_lora, lora_strength):
    key = (os.path.abspath(control_path), str(dtype),
           os.path.abspath(condition_lora) if condition_lora else None, round(float(lora_strength), 4))
    ctrl = _CACHE["control"].get(key)
    if ctrl is None:
        model = core.load_control_model(control_path, dtype=dtype,
                                        condition_lora=condition_lora, lora_strength=lora_strength)
        ctrl = core.LucidNFTControl(model)
        _CACHE["control"][key] = ctrl
    return ctrl


def _get_style(path):
    key = os.path.abspath(path)
    style = _CACHE["style"].get(key)
    if style is None:
        import comfy.ldm.flux.redux as redux
        model = redux.ReduxImageEncoder()
        model.load_state_dict(core.split_connector(path))
        style = comfy.sd.StyleModel(model)
        _CACHE["style"][key] = style
    return style


def _get_clip_vision(path):
    key = os.path.abspath(path)
    cv = _CACHE["clip_vision"].get(key)
    if cv is None:
        cv = comfy.clip_vision.load(path)
        _CACHE["clip_vision"][key] = cv
    return cv


# ---------------------------------------------------------------------------
# 节点
# ---------------------------------------------------------------------------

class LucidNFTLoRA:
    """LucidNFT 的 DiT LoRA（`lora_dit`，偏好优化后的 Dit 权重）+ 可选 turbo LoRA。

    官方给的是 PEFT adapter 目录，key 形如
    `base_model.model.double_blocks.0.img_attn.qkv.lora_A/B.weight`；
    这里改写成 comfy 通用格式交给 comfy 原生 LoRA patch（fp8 / 低显存都安全）。
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "dit_lora": (_adapter_list("dit"),
                             {"tooltip": "models/LucidFlux/lora_dit（PEFT adapter 目录）；none = 不打 LoRA"}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 4.0, "step": 0.05}),
            },
            "optional": {
                "extra_lora": (["none"] + folder_paths.get_filename_list("loras"),
                               {"tooltip": "可选：models/loras 里的标准 comfy LoRA（如 FLUX.1-Turbo-Alpha 加速）"}),
                "extra_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 4.0, "step": 0.05}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "LucidNFT"
    DESCRIPTION = ("LucidNFT 的 DiT LoRA（lora_dit）。条件分支的 LoRA（lora_condition）在 "
                   "LucidNFTCondition 里选，两边都接才是完整的 LucidNFT。")

    def apply(self, model, dit_lora, strength, extra_lora="none", extra_strength=1.0):
        path = _resolve_adapter(dit_lora)
        if path is not None and strength > 0:
            model = core.apply_dit_lora(model, path, strength)
        elif dit_lora not in (None, "none"):
            logger.warning("[LucidNFT] 没找到 adapter 目录: %s", dit_lora)

        if extra_lora not in (None, "none") and extra_strength > 0:
            p = folder_paths.get_full_path("loras", extra_lora)
            if p:
                model = core.apply_plain_lora(model, p, extra_strength)
        return (model,)


class LucidNFTDiffBIR:
    """DiffBIR 先验：SwinIR 通用复原（unshuffle 版本），在 latent 对应像素尺度上做一次“去退化”。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "swinir": (_list("swinir"),),
                "width": ("INT", {"default": 1024, "min": 0, "max": 8192, "step": 8}),
                "height": ("INT", {"default": 1024, "min": 0, "max": 8192, "step": 8}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("ldr", "lq")
    FUNCTION = "run"
    CATEGORY = "LucidNFT"
    DESCRIPTION = ("width/height = latent×8（latent 128 → 1024）；0 表示保持原尺寸。"
                   "lq = 缩放后的输入，ldr = SwinIR 复原结果，两者直接接 LucidNFTCondition。")

    def run(self, image, swinir, width, height):
        lq = _resize_bchw(_image_to_bchw(image), width, height)
        # 官方：condition 先 bf16 往返再 *2-1 -> ci_01
        cond = (lq.to(torch.bfloat16) * 2.0 - 1.0)
        ci01 = torch.clamp((cond.float() + 1.0) / 2.0, 0.0, 1.0)

        model = load_swinir(folder_paths.get_full_path(MODELS_SUBDIR, swinir))
        device = mm.get_torch_device()
        model.to(device)
        with torch.no_grad():
            out = model(ci01.to(device)).float().clamp(0.0, 1.0)
        model.to("cpu")
        mm.soft_empty_cache()

        ldr = _bchw_to_image(out)
        lq_img = _bchw_to_image(lq)
        if image.shape[0] == 1 and lq_img.shape[0] != image.shape[0]:
            lq_img = lq_img.repeat(image.shape[0], 1, 1, 1)
        return (ldr, lq_img)


class LucidNFTCondition:
    """一个节点搞定「文本 embedding + Redux 图像 embedding + 双条件分支（含 lora_condition）+ guidance」。

    - 文本：直接读官方 `prompt_embeddings.pt`（txt 1×512×4096 + vec 1×768），不加载 CLIP/T5；
      也可以把 embedding 设成 none，改用 `positive`（CLIPTextEncodeFlux 之类）的 txt/pooled；
    - 图像先验：SigLIP2-so400m-patch16-512(ldr) → connector(ReduxImageEncoder)
      → 1024×4096 token 追加到文本 token 之后（与官方一致）；
    - 条件分支：`lucidflux.pth` 的 condition_lq/ldr + modulation，可选合并 `lora_condition`，
      作为 comfy 原生 control 注入 flux double block（LQ/LDR 按 latent×8 自动对齐）。
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "control_file": (_list("lucid"), {"tooltip": "lucidflux.pth（双条件分支）"}),
                "condition_lora": (_adapter_list("condition"),
                                   {"tooltip": "models/LucidFlux/lora_condition（LucidNFT 的条件分支 LoRA）；none = 纯 LucidFlux 分支"}),
                "lora_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 4.0, "step": 0.05}),
                "embedding": (_list("prompt"), {"tooltip": "prompt_embeddings.pt（官方预计算文本 embedding）"}),
                "connector": (_list("connector"), {"tooltip": "lucid_connector.pth；也可直接给含 connector 段的 lucidflux.pth"}),
                "siglip": (_siglip_list(), {"tooltip": "models/clip_vision 下的 SigLIP2-512；接了 clip_vision 输入则忽略此项"}),
                "lq": ("IMAGE", {"tooltip": "低质输入（DiffBIR 的 lq，尺寸 = latent×8）"}),
                "ldr": ("IMAGE", {"tooltip": "DiffBIR 复原结果（SwinIR 输出）"}),
                "guidance": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "dtype": (["bf16", "fp32"], {"default": "bf16"}),
            },
            "optional": {
                "clip_vision": ("CLIP_VISION", {"tooltip": "可选：外部已加载的 SigLIP（省一次加载）"}),
                "positive": ("CONDITIONING", {"tooltip": "可选：embedding 设成 none 时，用这里的 txt/pooled 代替官方 pt"}),
            },
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "build"
    CATEGORY = "LucidNFT"
    DESCRIPTION = ("文本走官方 prompt_embeddings.pt（不加载 CLIP/T5）；SigLIP→Redux、"
                   "双条件分支（+lora_condition）、guidance 全部在这里完成，直接接 Sampler 的 positive。")

    def build(self, control_file, condition_lora, lora_strength, embedding, connector, siglip,
              lq, ldr, guidance, dtype, clip_vision=None, positive=None):
        dt = torch.bfloat16 if dtype == "bf16" else torch.float32

        # ---- 1. 文本 embedding（官方 pt，与 CLIP 推理等价）----
        if embedding not in (None, "none"):
            emb_path = folder_paths.get_full_path(MODELS_SUBDIR, embedding)
            if emb_path is None:
                raise ValueError(f"找不到 {embedding}（应放在 models/{MODELS_SUBDIR}/）")
            data = torch.load(emb_path, map_location="cpu", weights_only=False)
            txt = data["txt"].float()
            vec = data["vec"].float()
            if txt.ndim == 2:
                txt = txt.unsqueeze(0)
            if vec.ndim == 1:
                vec = vec.unsqueeze(0)
        elif positive is not None and len(positive) > 0:
            txt = positive[0][0].float()
            pooled = positive[0][1].get("pooled_output", None)
            if pooled is None:
                raise ValueError("positive 里没有 pooled_output，无法驱动 flux 的 vector_in")
            vec = pooled.float()
            if txt.ndim == 2:
                txt = txt.unsqueeze(0)
            if vec.ndim == 1:
                vec = vec.unsqueeze(0)
        else:
            raise ValueError("embedding 为 none 时必须接 positive")
        txt = txt.detach()

        # ---- 2. SigLIP → Redux，拼到文本 token 之后 ----
        cv = clip_vision
        if cv is None:
            cv_path = folder_paths.get_full_path("clip_vision", siglip)
            if cv_path is None:
                raise ValueError(f"找不到 SigLIP 权重: {siglip}")
            cv = _get_clip_vision(cv_path)
        cv_out = cv.encode_image(ldr, crop=False)          # 官方是拉伸 resize（不裁剪）
        if clip_vision is None:
            try:  # 编码完就卸下，不占着显存
                mm.unload_model_and_clones(cv.patcher)
                mm.soft_empty_cache()
            except Exception:
                pass
        connector_path = folder_paths.get_full_path(MODELS_SUBDIR, connector)
        if connector_path is None:
            raise ValueError(f"找不到 connector: {connector}")
        style = _get_style(connector_path)
        image_embeds = style.get_cond(cv_out).flatten(0, 1).unsqueeze(0)
        cond_txt = torch.cat((txt, image_embeds.to(txt.dtype)), dim=1)
        cond = [[cond_txt, {"pooled_output": vec}]]

        # ---- 3. 双条件分支（+ lora_condition）----
        control_path = folder_paths.get_full_path(MODELS_SUBDIR, control_file)
        if control_path is None:
            raise ValueError(f"找不到 {control_file}")
        ctrl = _get_control(control_path, dt, _resolve_adapter(condition_lora), lora_strength)

        lq_bchw = _image_to_bchw(lq)
        ldr_bchw = _image_to_bchw(ldr)
        if lq_bchw.shape[0] != ldr_bchw.shape[0]:
            ldr_bchw = comfy.utils.repeat_to_batch_size(ldr_bchw, lq_bchw.shape[0])
        applied = ctrl.copy_with(lq_bchw * 2.0 - 1.0, ldr_bchw * 2.0 - 1.0, txt)

        out = []
        for t in cond:
            d = t[1].copy()
            d["control"] = applied
            d["control_apply_to_uncond"] = False
            out.append([t[0], d])
        out = node_helpers.conditioning_set_values(out, {"guidance": float(guidance)})
        return (out,)


class LucidNFTSampler:
    """官方 get_schedule 的精确 sigma 表 + comfy 原生 euler。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "latent": ("LATENT",),
                "steps": ("INT", {"default": 20, "min": 1, "max": 200}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.01}),
                "shift": ("BOOLEAN", {"default": True,
                                      "tooltip": "官方 get_schedule 的 shift；flux-dev 开，flux-schnell 关"}),
            },
            "optional": {
                "noise": ("LATENT", {"tooltip": "仅在需要与官方逐位对拍时接入手工噪声"}),
            },
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"
    CATEGORY = "LucidNFT"
    DESCRIPTION = ("cfg 恒为 1.0（guidance 蒸馏模型，guidance 已在 LucidNFTCondition 里）。"
                   "steps 官方默认 24（dev）/ 4（配 turbo lora）。")

    def sample(self, model, positive, latent, steps, seed, cfg, shift, noise=None):
        latent_image = latent["samples"]
        sigmas = core.build_sigmas(steps, latent_image.shape[-2], latent_image.shape[-1], shift=shift)
        if noise is None:
            noise_samples = comfy.sample.prepare_noise(latent_image, seed)
        else:
            noise_samples = noise["samples"]
        samples = core.sample_lucidnft(model, positive, latent_image, sigmas,
                                       noise=noise_samples, seed=seed, cfg=cfg)
        return ({"samples": samples},)


class LucidNFTVAEPatch:
    """UltraFlux VAE 的 F16(/16) 语义补丁（comfy 默认按 /8 装载）。

    只在「用 UltraFlux.safetensors 出 2K」时需要；官方 LucidNFT 用 flux ae(/8)，
    那条路直接 VAELoader → VAEDecode，不需要这个节点。
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"vae": ("VAE",)}}

    RETURN_TYPES = ("VAE",)
    FUNCTION = "patch"
    CATEGORY = "LucidNFT"
    DESCRIPTION = ("可选：UltraFlux 的 F16 VAE 在官方是 /16。不接的话同一 latent 只解出一半分辨率。"
                   "用 flux ae 时不要接。")

    def patch(self, vae):
        return (core.patch_vae_f16(vae, stride=2),)


class LucidNFTColorAlign:
    """VAE 解码后的颜色对齐（官方 wavelet_reconstruction）。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "reference": ("IMAGE",),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "align"
    CATEGORY = "LucidNFT"
    DESCRIPTION = "把 reference（一般接 DiffBIR 的 ldr）的低频颜色/亮度搬回 image；strength=0 等于不处理。"

    def align(self, image, reference, strength):
        if strength <= 0.0:
            return (image,)
        x = _image_to_bchw(image)
        ref = _image_to_bchw(reference)
        if ref.shape[-2:] != x.shape[-2:]:
            ref = comfy.utils.common_upscale(ref, x.shape[-1], x.shape[-2], "bilinear", "disabled")
        if ref.shape[0] != x.shape[0]:
            ref = comfy.utils.repeat_to_batch_size(ref, x.shape[0])
        device = mm.get_torch_device()
        out = []
        with torch.no_grad():
            for i in range(x.shape[0]):
                fixed = wavelet_reconstruction(x[i:i + 1].to(device), ref[i:i + 1].to(device)).cpu()
                if strength < 1.0:
                    fixed = x[i:i + 1] * (1.0 - strength) + fixed * strength
                out.append(fixed)
        return (_bchw_to_image(torch.cat(out, dim=0)),)


NODE_CLASS_MAPPINGS = {
    "LucidNFTLoRA": LucidNFTLoRA,
    "LucidNFTDiffBIR": LucidNFTDiffBIR,
    "LucidNFTCondition": LucidNFTCondition,
    "LucidNFTSampler": LucidNFTSampler,
    "LucidNFTVAEPatch": LucidNFTVAEPatch,
    "LucidNFTColorAlign": LucidNFTColorAlign,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LucidNFTLoRA": "LucidNFT LoRA (dit)",
    "LucidNFTDiffBIR": "LucidNFT DiffBIR (SwinIR)",
    "LucidNFTCondition": "LucidNFT Condition (text+redux+control+lora)",
    "LucidNFTSampler": "LucidNFT Sampler",
    "LucidNFTVAEPatch": "LucidNFT VAE Patch (F16, optional)",
    "LucidNFTColorAlign": "LucidNFT Color Align",
}
