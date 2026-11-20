"""Download the open-source head-swap EDIT models (Qwen-Image-Edit-2511 + BFS head v5 + Lightning) into the bfs-models volume.
Run once:  modal run download_edit_models.py::download"""
import os
import modal

app = modal.App("bfs-download-edit-models")
vol = modal.Volume.from_name("bfs-models", create_if_missing=True)
MODELS = "/models"
image = modal.Image.debian_slim(python_version="3.11").pip_install("huggingface_hub[hf_transfer]").env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})

FILES = [
    # Qwen-Image-Edit-2511 (Apache-2.0), int8 like the Qwen 2.1 model that already runs here
    ("Comfy-Org/Qwen-Image-Edit_ComfyUI", "split_files/diffusion_models/qwen_image_edit_2511_int8_convrot.safetensors", "diffusion_models/qwen_image_edit_2511_int8_convrot.safetensors"),
    ("Comfy-Org/Qwen-Image_ComfyUI", "split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors", "text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors"),
    ("Comfy-Org/Qwen-Image_ComfyUI", "split_files/vae/qwen_image_vae.safetensors", "vae/qwen_image_vae.safetensors"),
    # 8-step speed LoRA for 2511
    ("lightx2v/Qwen-Image-Edit-2511-Lightning", "Qwen-Image-Edit-2511-Lightning-8steps-V1.0-bf16.safetensors", "loras/Qwen-Image-Edit-2511-Lightning-8steps-V1.0-bf16.safetensors"),
    # BFS head swap V5 for 2511 (merged rank-16: the author's practical default)
    ("Alissonerdx/BFS-Best-Face-Swap", "bfs_head_v5_2511_merged_version_rank_16_fp16.safetensors", "loras/bfs_head_v5_2511_merged_version_rank_16_fp16.safetensors"),
]


@app.function(image=image, volumes={MODELS: vol}, timeout=3600)
def download():
    import shutil
    from huggingface_hub import hf_hub_download
    for repo, path, dest in FILES:
        out = os.path.join(MODELS, dest)
        if os.path.exists(out):
            print("have", dest)
            continue
        p = hf_hub_download(repo, path)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        shutil.copyfile(p, out)
        print("got", dest, os.path.getsize(out))
    vol.commit()
