 # !/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np
import torch
import os
from einops import rearrange
from .model_loader_utils import  clear_comfyui_cache,nomarl_upscale,gc_cleanup
from .inference import pre_img,get_cond
from .src.flux.lucidflux import load_connector,apply_lora,load_checkpoint_bundle
from .src.flux.sampling import denoise_lucidflux, get_schedule,unpack
from .src.flux.util import load_ae
import folder_paths
from typing_extensions import override
from comfy_api.latest import ComfyExtension, io
import nodes
import comfy.model_management as mm
from .src.flux.align_color import wavelet_reconstruction
from torchvision.utils import save_image
MAX_SEED = np.iinfo(np.int32).max

device = torch.device(
    "cuda:0") if torch.cuda.is_available() else torch.device(
    "mps") if torch.backends.mps.is_available() else torch.device("cpu")

node_cr_path = os.path.dirname(os.path.abspath(__file__))

weigths_LucidFlux_current_path = os.path.join(folder_paths.models_dir, "LucidFlux")
if not os.path.exists(weigths_LucidFlux_current_path):
    os.makedirs(weigths_LucidFlux_current_path)
folder_paths.add_model_folder_path("LucidFlux", weigths_LucidFlux_current_path) #  LucidFlux dir

class LucidNFT_SM_Model(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        
        return io.Schema(
            node_id="LucidNFT_SM_Model",
            display_name="LucidNFT_SM_Model",
            category="LucidNFT",
            inputs=[
                io.Combo.Input("LucidFlux",options= ["none"] + [i for i in folder_paths.get_filename_list("LucidFlux") if "lucid" in i.lower()]),
                io.Combo.Input("diffusion_models",options= ["none"] + folder_paths.get_filename_list("diffusion_models")),
                io.Boolean.Input("block_offload", default=True),
                io.Combo.Input("model_type",options= ["bf16","f32"] ),
                io.Model.Input("cf_model", optional=True),
                
            ],
            outputs=[
                io.Model.Output(display_name="model"),
                ],
            )
    @classmethod
    def execute(cls, LucidFlux,diffusion_models,block_offload,model_type,cf_model=None) -> io.NodeOutput:
        model_dtype = torch.bfloat16 if model_type == 'bf16' else torch.float32
        name="flux-dev" if "dev" in diffusion_models.lower() else "flux-schnell"
        if cf_model is not None:    
            name="flux-dev" if "guidance_in.in_layer.weight" in cf_model.model.diffusion_model.state_dict().keys() else "flux-schnell"
            print("flux is :",name)
        LucidFlux_path=folder_paths.get_full_path("LucidFlux", LucidFlux) if LucidFlux != "none" else None
        ckpt_path=folder_paths.get_full_path("diffusion_models", diffusion_models) if diffusion_models != "none" else None
        assert LucidFlux_path is not None,"need LucidFlux"
        args={
            "name":name,
            "checkpoint_path":LucidFlux_path,
            "torch_device":device,
            "model_dtype":model_dtype,
            "offload":block_offload,
            "ckpt_path":ckpt_path,
            "cf_model":cf_model,
            "node_cr_path":node_cr_path,
        }
        model,dual_condition_branch=load_checkpoint_bundle(**args)
        model.block_offload=block_offload
        model.is_schnell=name=="flux-schnell"
        return io.NodeOutput({"model": model,  "dual_condition_branch": dual_condition_branch, })
    

class LucidNFT_SM_Diffbir(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LucidNFT_SM_Diffbir",
            display_name="LucidNFT_SM_Diffbir",
            category="LucidNFT",
            inputs=[
                io.Combo.Input("swinir",options= ["none"] + folder_paths.get_filename_list("LucidFlux") ),
                io.Image.Input("image"),
                io.Int.Input("width", default=1024, min=256, max=nodes.MAX_RESOLUTION,step=64,display_mode=io.NumberDisplay.number),
                io.Int.Input("height", default=1024, min=256, max=nodes.MAX_RESOLUTION,step=64,display_mode=io.NumberDisplay.number),
            ],
            outputs=[
                io.Conditioning.Output(display_name="cond"),
                io.Image.Output(display_name="image"),
                ],
            )
    @classmethod
    def execute(cls,swinir, image,width,height) -> io.NodeOutput:
        clear_comfyui_cache()   
        swinir_path=folder_paths.get_full_path("LucidFlux", swinir) if swinir != "none" else None
        assert swinir_path is not None,"need swinir"
        image,condition_cond_lq,condition_cond_ldr=pre_img(swinir_path,image, width,height,torch.float32,device) # keep image in float32
        cond={"image":image,"condition_cond_lq":condition_cond_lq,"condition_cond_ldr":condition_cond_ldr}
        return io.NodeOutput(cond,image)


class LucidNFT_SM_Cond(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LucidNFT_SM_Cond",
            display_name="LucidNFT_SM_Cond",
            category="LucidNFT",
            inputs=[
                io.Model.Input("model"),
                io.Combo.Input("turbo_lora",options= ["none"] + folder_paths.get_filename_list("loras")),
                io.Float.Input("scale", default=1.0, min=0.0, max=1.0, step=0.1,display_mode=io.NumberDisplay.number),
                ],
            outputs=[io.Model.Output(display_name="model")],
        )
    @classmethod
    def execute(cls, model,turbo_lora,scale) -> io.NodeOutput:
        turbo_lora_path=folder_paths.get_full_path("loras", turbo_lora) if turbo_lora!="none" else None
        model=apply_lora(model,weigths_LucidFlux_current_path,device,turbo_lora_path,scale)
        return io.NodeOutput (model)


class LucidNFT_SM_Encode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LucidNFT_SM_Encode",
            display_name="LucidNFT_SM_Encode",
            category="LucidNFT",
            inputs=[
                io.ClipVision.Input("CLIP_VISION"),
                io.Conditioning.Input("cond"),#  B H W C C=3
                io.Int.Input("seed", default=0, min=0, max=MAX_SEED,display_mode=io.NumberDisplay.number),
                io.Combo.Input("emb",options= ["none"] + [i for i in folder_paths.get_filename_list("LucidFlux") if "prompt" in i.lower() ]),
                io.Combo.Input("connector",options= ["none"] + [i for i in folder_paths.get_filename_list("LucidFlux") if "connector" in i.lower() ]),
                io.Combo.Input("model_type",options= ["bf16","f32"] ),
                io.Conditioning.Input("positive",optional=True),     
            ],
            outputs=[
                io.Conditioning.Output(display_name="condition"),
                ],
        )
    @classmethod
    def execute(cls, CLIP_VISION, cond,seed,emb,connector,model_type,positive=None) -> io.NodeOutput:
        image=cond["image"]
        model_dtype = torch.bfloat16 if model_type == 'bf16' else torch.float32
        emb_path=folder_paths.get_full_path("LucidFlux", emb) if emb != "none" else None
        _,height,width,_=image.shape
        if width % 16 != 0 or height % 16 !=0:
            width = 16 * (width // 16)
            height = 16 * (height // 16)
            image=nomarl_upscale(image, width,height)
        torch.manual_seed(seed)

        # embedding
        inp_cond=get_cond(positive,emb_path,height,width,device,model_dtype,seed) 

        # siglip
        siglip_image_pre_fts=CLIP_VISION.encode_image(image)["last_hidden_state"].to(device=device,dtype=model_dtype) 
        clear_comfyui_cache()

        # connector
        connector_path=folder_paths.get_full_path("LucidFlux", connector) if connector != "none" else None
        assert connector_path is not None,"need connector"
        connector_source=torch.load(connector_path, map_location="cpu",weights_only=True) 
        connector = load_connector("cpu" , model_dtype, connector_source)
        connector_dtype = connector.redux_up.weight.dtype
        connector.to(device)
        image_embeds = connector(siglip_image_pre_fts.to(device=device, dtype=connector_dtype))["image_embeds"]
        connector.to("cpu")
        gc_cleanup()
        
        txt = inp_cond["txt"].to(device=device, dtype=model_dtype)
        txt_ids = inp_cond["txt_ids"].to(device=device, dtype=model_dtype)
        siglip_txt = torch.cat([txt, image_embeds.to(dtype=model_dtype)], dim=1)
        batch_size, _, channels = txt_ids.shape
        extra_ids = torch.zeros((batch_size, 1024, channels), device=txt_ids.device, dtype=model_dtype)
        siglip_txt_ids = torch.cat([txt_ids, extra_ids], dim=1).to(dtype=model_dtype)
        condition={"txt":txt,"txt_ids":txt_ids,"siglip_txt":siglip_txt,"siglip_txt_ids":siglip_txt_ids,"width":width,"height":height,"image":cond["image"],
                   "vec":inp_cond["vec"],"img_ids":inp_cond["img_ids"],"img":inp_cond["img"],"condition_cond_lq":cond["condition_cond_lq"],"condition_cond_ldr":cond["condition_cond_ldr"]
        }
        
        return io.NodeOutput(condition)


class LucidNFT_SM_KSampler(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LucidNFT_SM_KSampler",
            display_name="LucidNFT_SM_KSampler",
            category="LucidNFT",
            inputs=[
                io.Model.Input("model"),
                io.Conditioning.Input("condition"),
                io.Int.Input("steps", default=20, min=1, max=10000,display_mode=io.NumberDisplay.number),
                io.Float.Input("cfg", default=4.0, min=0.0, max=100.0, step=0.1, round=0.01,display_mode=io.NumberDisplay.number),
            ],
            outputs=[
                io.Latent.Output(display_name="latent"),
            ],
        )
    @classmethod
    def execute(cls, model,condition,steps, cfg) -> io.NodeOutput:
        dual_condition_model=model["dual_condition_branch"].to(device)
        height, width = condition["height"], condition["width"]
        model=model["model"]
        if not  model.block_offload:
            model.to(device)
        condition["guidance"]=cfg
        condition["timesteps"]=get_schedule(steps,(width // 8) * (height // 8) // (16 * 16),shift=(not model.is_schnell),)
        # denoise
        with torch.no_grad():
            x=denoise_lucidflux(model,dual_condition_model,**condition)
            x = unpack(x.float(), height, width)
        output={"samples":x,"image":condition["image"]}
        return io.NodeOutput(output)
    
class LucidNFT_SM_Decoder(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LucidNFT_SM_Decoder",
            display_name="LucidNFT_SM_Decoder",
            category="LucidNFT",
            inputs=[ 
                io.Combo.Input("vae",options= ["none"] + folder_paths.get_filename_list("vae")),
                io.Latent.Input("latent"),
                io.Boolean.Input("wavelet", default=True),
                io.Vae.Input("ae",optional=True),
            ],
            outputs=[
                io.Image.Output(display_name="image"),
            ],
        )
    @classmethod
    def execute(cls,vae, latent,wavelet,ae=None,) -> io.NodeOutput:
        x=latent["samples"]
        vae_path=folder_paths.get_full_path("vae", vae) if vae != "none" else None
        if ae is not None:
            x=(x/0.3611)+0.1159  # add mean
            x=ae.decode(x) #torch.Size([1, 1024, 1024, 3])
            clear_comfyui_cache()
            if wavelet:
                x1 = x.permute(0, 3, 1, 2) #--> torch.Size([1, 3, 1024, 1024])
                x1 = rearrange(x1[-1], "c h w -> h w c").to("cpu")
                x1 = wavelet_reconstruction(x1.permute(2, 0, 1), latent["image"].permute(0, 3, 1, 2).squeeze(0).to("cpu"))
                x1 = x1.clamp(0, 1)
                img=x1.unsqueeze(0).permute(0, 2, 3, 1) 
            else:
                img = x
        elif vae_path is not None:
            ae = load_ae("flux-dev", device,vae_path)
            x = ae.decode(x) #torch.Size([1, 3, 1024, 1024])
            if wavelet:
                x1 = x.clamp(-1, 1)
                x1 = rearrange(x1[-1], "c h w -> h w c").to("cpu")
                hq = wavelet_reconstruction((x1.permute(2, 0, 1) + 1.0) / 2, latent["image"].permute(0, 3, 1, 2).squeeze(0).to("cpu"))
                hq = hq.clamp(0, 1)
                save_image(hq, os.path.join(folder_paths.get_output_directory(), f"{123}.png"))
                img=hq.unsqueeze(0).permute(0, 2, 3, 1)
            else:
                img =((x +1.0)/2).clamp(0, 1).permute(0, 2, 3, 1)
        else: 
            raise NotImplementedError("vae")
        return io.NodeOutput(img)

class LucidNFT_SM_Extension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            LucidNFT_SM_Model,
            LucidNFT_SM_Diffbir,
            LucidNFT_SM_Cond,
            LucidNFT_SM_Encode,
            LucidNFT_SM_KSampler,
            LucidNFT_SM_Decoder,
        ]

async def comfy_entrypoint() -> LucidNFT_SM_Extension:  # ComfyUI calls this to load your extension and its nodes.
    return LucidNFT_SM_Extension()
