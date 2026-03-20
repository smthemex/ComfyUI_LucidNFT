# ComfyUI_LucidNFT
[LucidNFT](https://github.com/W2GenAI-Lab/LucidNFT):LR-Anchored Multi-Reward Preference Optimization for Generative Real-World Super-Resolution

# Update
* 解码部分推荐用模型，comfyUI的ae解码流程（连标准vae节点时）会有一定的色差，记得给官方[LucidNFT](https://github.com/W2GenAI-Lab/LucidNFT)点星;  
* It is recommended to use a model for the decoding part. The ae decoding process in ComfyUI (when connected to the standard VAE node) will have some color differences.Start [LucidNFT](https://github.com/W2GenAI-Lab/LucidNFT) if you like it.

1.Installation  
-----
  In the ./ComfyUI/custom_nodes directory, run the following:   
```
git clone https://github.com/smthemex/ComfyUI_LucidNFT
```
2.requirements  
----
```
pip install -r requirements.txt
```

3.checkpoints 
----
* Any normal flux dit / 任意标准flux模型， KJ的或者官方封装的
* Lucid checkpoints [links](https://huggingface.co/W2GenAI/LucidFlux/tree/main) /lucidflux.pth and prompt_embeddings.pt
* Lucid lroa  [links](https://huggingface.co/W2GenAI/LucidNFT/tree/main)
* Siglip512 [links](https://huggingface.co/google/siglip2-so400m-patch16-512/tree/main) / model.safetensors 只下单体模型   
* DiffBIR [links](https://huggingface.co/lxq007/DiffBIR/tree/main)  /  general_swinir_v1.ckpt   
* Turbo lora [links](https://huggingface.co/alimama-creative/FLUX.1-Turbo-Alpha)  #optional 可选，8 步起  
*  Flux ae   [links](https://huggingface.co/Comfy-Org/models)   
* Connercter [links](https://huggingface.co/smthem/LucidFLUX-connector)
```
├── ComfyUI/models/
|     ├── diffusion_models/any flux dit # 任意flux dit模型 ，就用kj的或者x flux的，名字要带dev 否则跑schnell
|     ├── vae/ae.safetensors #comfy 
|     ├── clip_vision/siglip2-so400m-patch16-512.safetensors  #rename from model.safetensors  最好重命名个，不然都是siglip 的model.safetensors
|     ├── LucidFlux/
|        ├──general_swinir_v1.ckpt
|        ├──lucidflux.pth
|        ├──prompt_embeddings.pt # 已适配，使用时不要连clip
|        ├──lucid_connector.pth # split from lucidflux.pth
|        ├── lora_condition/
|             ├──adapter_config.json
|             ├──adapter_model.safetensors
|        ├── lora_dit/
|             ├──adapter_config.json
|             ├──adapter_model.safetensors
```


# 4 .Example
![](https://github.com/smthemex/ComfyUI_LucidNFT/blob/main/example_workflows/example2.png)
![](https://github.com/smthemex/ComfyUI_LucidNFT/blob/main/example_workflows/example.png)


# 5. Citation
```
@article{fei2026lucidnft,
  title={LucidNFT: LR-Anchored Multi-Reward Preference Optimization for Generative Real-World Super-Resolution},
  author={Fei, Song and Ye, Tian and Chen, Sixiang and Xing, Zhaohu and Lai, Jianyu and Zhu, Lei},
  journal={arXiv preprint arXiv:2603.05947},
  year={2026}
}
``
