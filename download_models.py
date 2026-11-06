"""One-off: download the model files into a Modal Volume ("bfs-models").

Run once per Modal workspace:   modal run download_models.py::download
"""
import os
import modal

app = modal.App("bfs-download-models")
vol = modal.Volume.from_name("bfs-models", create_if_missing=True)
MODELS = "/models"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("huggingface_hub[hf_transfer]")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

# (hugging-face repo, file in repo, destination inside the volume)
FILES = [
    ("Comfy-Org/Qwen-Image-2.1", "diffusion_models/qwen_image_2.1_int8_convrot.safetensors", "diffusion_models/qwen_image_2.1_int8_convrot.safetensors"),
    ("Comfy-Org/Qwen-Image-2.1", "text_encoders/qwen3vl_8b_int8_convrot.safetensors", "text_encoders/qwen3vl_8b_int8_convrot.safetensors"),
    ("Comfy-Org/Qwen-Image-2.1", "vae/qwen_image_2.1_vae_bf16.safetensors", "vae/qwen_image_2.1_vae_bf16.safetensors"),
    ("Comfy-Org/Qwen-Image-2.1", "loras/p_q_image_2.1_8_steps_v0.1_comfyui_bf16.safetensors", "loras/p_q_image_2.1_8_steps_v0.1_comfyui_bf16.safetensors"),
    ("Alissonerdx/BFS-Best-Face-Swap", "bfs_head_v1.1_qwen_2.1.safetensors", "loras/bfs_head_v1.1_qwen_2.1.safetensors"),
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
