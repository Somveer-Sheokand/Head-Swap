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
    # temporal refiner ("ultra" quality): Wan 2.2 TI2V-5B, Apache-2.0
    ("Comfy-Org/Wan_2.2_ComfyUI_Repackaged", "split_files/diffusion_models/wan2.2_ti2v_5B_fp16.safetensors", "diffusion_models/wan2.2_ti2v_5B_fp16.safetensors"),
    ("Comfy-Org/Wan_2.2_ComfyUI_Repackaged", "split_files/vae/wan2.2_vae.safetensors", "vae/wan2.2_vae.safetensors"),
    ("Comfy-Org/Wan_2.2_ComfyUI_Repackaged", "split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors", "text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors"),
    # video-native head replacement ("animate"): Wan 2.2 Animate 14B (Apache-2.0) + relight LoRA + 4-6 step LoRA
    ("Comfy-Org/Wan_2.2_ComfyUI_Repackaged", "split_files/diffusion_models/wan2.2_animate_14B_int8_convrot.safetensors", "diffusion_models/wan2.2_animate_14B_int8_convrot.safetensors"),
    ("Comfy-Org/Wan_2.2_ComfyUI_Repackaged", "split_files/loras/wan2.2_animate_14B_relight_lora_bf16.safetensors", "loras/wan2.2_animate_14B_relight_lora_bf16.safetensors"),
    ("Kijai/WanVideo_comfy", "Lightx2v/lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors", "loras/lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors"),
    ("Comfy-Org/Wan_2.2_ComfyUI_Repackaged", "split_files/vae/wan_2.1_vae.safetensors", "vae/wan_2.1_vae.safetensors"),
    ("Comfy-Org/Wan_2.1_ComfyUI_repackaged", "split_files/clip_vision/clip_vision_h.safetensors", "clip_vision/clip_vision_h.safetensors"),
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
