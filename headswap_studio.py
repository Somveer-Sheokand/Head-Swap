"""HEADSWAP STUDIO - best-quality image head swap (deepak-sheokand Modal account).
Main endpoint: https://deepak-sheokand--headswap-studio-api.us-east.modal.run  (POST /headswap, GET /jobs/{id}, GET /download/{path})
Pipeline: Qwen Image 2.1 + BFS head v1.1 (or Qwen-Image-Edit-2511 + BFS head V5) -> best-of-N selection -> head+neck pasted onto
the ORIGINAL image -> clothes/jewellery protection, old-hair removal, background match -> two-zone face/neck skin-tone match ->
everything outside the head/neck restored bit-exact.
"""
"""BFS head-swap service on Modal.

Model: Qwen Image 2.1 + BFS LoRA bfs_head_v1.1_qwen_2.1 (Alissonerdx/BFS-Best-Face-Swap) via ComfyUI.
Pipeline: detect target head per frame (InsightFace) -> crop head -> BFS on crop -> feathered paste-back
          + colour match -> re-encode with original audio.
"""
import json, os, subprocess, time, urllib.request, urllib.parse, uuid, io
import modal

app = modal.App("headswap-studio")
models_vol = modal.Volume.from_name("bfs-models", create_if_missing=True)
jobs_vol = modal.Volume.from_name("bfs-jobs", create_if_missing=True)
MODELS, JOBS = "/models", "/jobs"
GPU_USD_PER_SEC = 1.95 / 3600  # L40S, approx
# quality presets used by every endpoint ("best" = previous behaviour)
PRESETS = {
    # post_*: second motion-aligned temporal pass over the finished heads; denoise: light temporal denoise (ffmpeg
    # hqdn3d) at the join. Tested on a 15 s close-up ad: head flicker 7.98 -> 6.87 (original footage 7.10), no ghosting
    # stab_*: head stabiliser (each generated head registered onto its motion-compensated neighbours, 2 passes);
    # cm_smooth: colour-match parameters smoothed over +-2 frames. Same ad: added wobble -14 % shift / -20 % scale
    "best": dict(res=1024, href=768, key_step=1, taps=2, post_taps=3, post_w=0.8, post_thr=16.0, denoise=True,
                 stab_iters=2, stab_taps=2, stab_w=0.8, cm_smooth=2.0),
    # adaptive: small / far-away heads are generated at a smaller size (never below MIN_RES) - they are shrunk
    # to their on-screen size anyway, so the extra pixels cost GPU time without being visible
    "fast": dict(res=832, href=640, key_step=2, taps=2, adaptive=True),
    # eco: same look as best, but BFS heads only on keyframes (every 8th frame + frames where RAFT-warping fails,
    # ~40-55 % fewer GPU frames); in-between frames carry the key heads along the original motion. Least wobble,
    # but faint seams can appear in very fast head turns - use for calm footage
    "eco": dict(res=1024, href=768, key_step=8, prop=True, flow="raft", prop_thr=5.5, prop_gate=12.0, taps=2,
                post_taps=3, post_w=0.8, post_thr=16.0, denoise=True, stab_iters=2, stab_taps=2, stab_w=0.8, cm_smooth=2.0),
}
MIN_RES = 640
EST_FACTOR = 2.0                     # real Modal bills were ~2x the plain per-frame estimate (cold starts, idle GPU time)

# Tested and left OFF: SageAttention made a 12-frame batch 9 % SLOWER (194.9 s vs 179.1 s); container start-up is
# only ~10 s, so a GPU snapshot would save cents. 6 instead of 8 steps saves only 7 % (166 s vs 179 s).
USE_SNAPSHOT = False
USE_SAGE = False
GPU_PARALLEL = 8                     # GPU containers per video (same total GPU-seconds, ~half the wall time of 4)
BLEND_CHUNK, BLEND_PAD = 24, 10      # frames per parallel blend container, overlap frames each side
KEEP_DAYS = 14                       # outputs / uploads older than this are deleted automatically

gpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "ffmpeg", "libgl1", "libglib2.0-0")
    .pip_install("torch==2.8.0", "torchvision", "torchaudio", index_url="https://download.pytorch.org/whl/cu128")
    .run_commands(
        "git clone --depth 1 https://github.com/comfyanonymous/ComfyUI /comfy",
        "cd /comfy && pip install -r requirements.txt",
        "pip install huggingface_hub websocket-client",
    )
)
if USE_SAGE:   # only changes the image when switched on (keeps torch untouched otherwise)
    gpu_image = gpu_image.pip_install("triton", "sageattention==1.0.6")

cpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libgl1", "libglib2.0-0", "build-essential")
    .pip_install("numpy<2", "opencv-python-headless==4.10.0.84", "onnxruntime==1.20.1", "insightface==0.7.3",
                 "requests", "fastapi[standard]")
    .run_commands("python -c \"from insightface.app import FaceAnalysis; "
                  "a=FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'], allowed_modules=['detection','recognition']); "
                  "a.prepare(ctx_id=-1, det_size=(640,640))\"")
)

cpu_image = cpu_image.pip_install("torch==2.5.1", "torchvision==0.20.1", index_url="https://download.pytorch.org/whl/cpu").run_commands(
    "python -c \"from torchvision.models.optical_flow import raft_large, Raft_Large_Weights; raft_large(weights=Raft_Large_Weights.DEFAULT)\"",
    "mkdir -p /seg",
    "python -c \"import urllib.request as u; b='https://huggingface.co/jonathandinu/face-parsing/resolve/main/onnx/'; [u.urlretrieve(b+f, '/seg/'+f) for f in ('model_quantized.onnx','model.onnx')]\"",
)

PROMPT = ("head_swap: start with <image1> as the base image, keeping its lighting, environment, and background. "
          "remove the head from <image1> completely and replace it with the head from <image2>, strictly preserving "
          "the hair, eye color, nose structure from <image2>. copy the direction of the eye, head rotation, micro "
          "expressions from <image1>, high quality, sharp details, 4k")
BODY_RES, HEAD_RES = 768, 640


# ----------------------------------------------------------------------------- GPU worker (ComfyUI)
PROMPT3 = PROMPT + (". <image3> is the previous video frame already swapped: keep the new head looking exactly as in <image3> (same skin texture, hair strands, lighting and sharpness), only adapting the pose and expression from <image1>")


def _graph(body, head, seed, steps, sampler, res=BODY_RES, href=HEAD_RES, prev=None, vae_up=False, out_res=0, cfg=1.0, distill=True):
    g = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "qwen_image_2.1_int8_convrot.safetensors", "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen3vl_8b_int8_convrot.safetensors", "type": "qwen_image", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": "qwen_image_2.1_vae_bf16.safetensors"}},
        "4": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["1", 0], "lora_name": "p_q_image_2.1_8_steps_v0.1_comfyui_bf16.safetensors", "strength_model": 1.0}},
        "5": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["4", 0], "lora_name": "bfs_head_v1.1_qwen_2.1.safetensors", "strength_model": 1.0}},
        "6": {"class_type": "QwenImage21Cache", "inputs": {"model": ["5", 0], "device": "auto", "dtype": "default"}},
        "7": {"class_type": "LoadImage", "inputs": {"image": body}},
        "8": {"class_type": "LoadImage", "inputs": {"image": head}},
        "9": {"class_type": "ImageScale", "inputs": {"image": ["7", 0], "upscale_method": "lanczos", "width": res, "height": res, "crop": "disabled"}},
        "10": {"class_type": "ImageScale", "inputs": {"image": ["8", 0], "upscale_method": "lanczos", "width": href, "height": href, "crop": "disabled"}},
        "11": {"class_type": "TextEncodeQwenImage21", "inputs": {"clip": ["2", 0], "prompt": PROMPT, "negative_prompt": "", "resolution": 0,
                                                                  "images.image_1": ["9", 0], "images.image_2": ["10", 0], "vae": ["3", 0]}},
        "12": {"class_type": "EmptyLatentImage", "inputs": {"width": res, "height": res, "batch_size": 1}},
        "13": {"class_type": "KSampler", "inputs": {"model": ["6", 0], "seed": seed, "steps": steps, "cfg": 1.0, "sampler_name": sampler,
                                                      "scheduler": "simple", "positive": ["11", 0], "negative": ["11", 1], "latent_image": ["12", 0], "denoise": 1.0}},
        "14": {"class_type": "VAEDecode", "inputs": {"samples": ["13", 0], "vae": ["3", 0]}},
        "15": {"class_type": "SaveImage", "inputs": {"images": ["14", 0], "filename_prefix": "out"}},
    }
    if not distill:
        # full model (no 8-step speed LoRA): more steps + real guidance -> more natural skin, hair and light, ~4x slower
        g["5"]["inputs"]["model"] = ["1", 0]
        del g["4"]
        g["13"]["inputs"]["cfg"] = cfg
    # NOTE: the Wan2.1 2x-upscale VAE from the user workflow cannot be used here - Qwen Image 2.1 latents have 64
    # channels, that VAE (and Qwen-Image-Edit / Wan2.1) uses 16 (checked in the safetensors headers)
    if out_res and out_res != res:
        o = out_res or res
        g["31"] = {"class_type": "ImageScale", "inputs": {"image": ["14", 0], "upscale_method": "lanczos", "width": o, "height": o, "crop": "disabled"}}
        g["15"]["inputs"]["images"] = ["31", 0]
    if prev:
        g["16"] = {"class_type": "LoadImage", "inputs": {"image": prev}}
        g["17"] = {"class_type": "ImageScale", "inputs": {"image": ["16", 0], "upscale_method": "lanczos", "width": res, "height": res, "crop": "disabled"}}
        g["11"]["inputs"]["images.image_3"] = ["17", 0]
        g["11"]["inputs"]["prompt"] = PROMPT3
    return g


# ----------------------------------------------------------------------------- PRO engine: Qwen-Image-Edit-2511 + BFS head V5
# Open-source (Apache-2.0) instruction image editor. Input order for BFS V5: Picture 1 = target body, Picture 2 = new face.
# Official BFS V5 prompt + our own instructions (learned from the ad work: keep jewellery/clothes, natural lips, skin match).
EDIT_PROMPT = ("head_swap: start with Picture 1 as the base image, keeping its lighting, environment, and background. remove the head "
               "from Picture 1 completely and replace it with the head from Picture 2, strictly preserving the hair, eye color, and nose "
               "structure of Picture 2. copy the eye direction, head rotation, and micro-expressions from Picture 1.")
EDIT_PROMPT_EXTRA = (" Keep the clothes, collar and any jewellery of Picture 1 exactly as they are and do not add any new jewellery. "
                     "Soft natural photographic skin texture with no over-sharpening, the skin tone of the face and neck matches the body "
                     "of Picture 1, seamless natural jaw and neck, natural hairline, no glasses, no lipstick.")


def _edit_graph(body, head, seed, steps, res=1024, href=1024, prompt=EDIT_PROMPT, lightning=True, lora=1.0, cfg=1.0, shift=3.0):
    g = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "qwen_image_edit_2511_int8_convrot.safetensors", "weight_dtype": "default"}},
        "2": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["1", 0], "lora_name": "Qwen-Image-Edit-2511-Lightning-8steps-V1.0-bf16.safetensors", "strength_model": 1.0}},
        "3": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["2", 0], "lora_name": "bfs_head_v5_2511_merged_version_rank_16_fp16.safetensors", "strength_model": lora}},
        "4": {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": ["3", 0], "shift": shift}},
        "5": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen_2.5_vl_7b_fp8_scaled.safetensors", "type": "qwen_image", "device": "default"}},
        "6": {"class_type": "VAELoader", "inputs": {"vae_name": "qwen_image_vae.safetensors"}},
        "7": {"class_type": "LoadImage", "inputs": {"image": body}},
        "8": {"class_type": "LoadImage", "inputs": {"image": head}},
        "9": {"class_type": "ImageScale", "inputs": {"image": ["7", 0], "upscale_method": "lanczos", "width": res, "height": res, "crop": "disabled"}},
        "10": {"class_type": "ImageScale", "inputs": {"image": ["8", 0], "upscale_method": "lanczos", "width": href, "height": href, "crop": "disabled"}},
        "11": {"class_type": "TextEncodeQwenImageEditPlus", "inputs": {"clip": ["5", 0], "prompt": prompt, "vae": ["6", 0], "image1": ["9", 0], "image2": ["10", 0]}},
        "12": {"class_type": "TextEncodeQwenImageEditPlus", "inputs": {"clip": ["5", 0], "prompt": "", "vae": ["6", 0], "image1": ["9", 0], "image2": ["10", 0]}},
        "13": {"class_type": "VAEEncode", "inputs": {"pixels": ["9", 0], "vae": ["6", 0]}},
        "14": {"class_type": "KSampler", "inputs": {"model": ["4", 0], "seed": seed, "steps": steps, "cfg": cfg, "sampler_name": "euler",
                                                      "scheduler": "simple", "positive": ["11", 0], "negative": ["12", 0], "latent_image": ["13", 0], "denoise": 1.0}},
        "15": {"class_type": "VAEDecode", "inputs": {"samples": ["14", 0], "vae": ["6", 0]}},
        "16": {"class_type": "SaveImage", "inputs": {"images": ["15", 0], "filename_prefix": "edit"}},
    }
    if not lightning:
        g["3"]["inputs"]["model"] = ["1", 0]
        del g["2"]
    return g


# ----------------------------------------------------------------------------- KLEIN engine: Flux 2 Klein 4B + BFS head (Apache-2.0)
# Most photographic engine in our tests (natural skin/hair, keeps the head angle). Picture 1 = body crop, Picture 2 = head.
KLEIN_PROMPT = ("head_swap: start with Picture 1 as the base image, keeping its lighting, environment, and background. "
                "remove the head from Picture 1 completely and replace it with the head from Picture 2, strictly preserving "
                "the hair, eye color, and nose structure of Picture 2. copy the direction of the eye, head rotation, and "
                "micro-expressions from Picture 1. Keep the body, clothing, pose, framing, and background intact.")
klein_vol = modal.Volume.from_name("klein-hf-cache", create_if_missing=True)
klein_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "curl")
    .pip_install("torch==2.8.0", "torchvision", index_url="https://download.pytorch.org/whl/cu128")
    .pip_install("git+https://github.com/huggingface/diffusers.git", "transformers>=4.57", "accelerate", "peft>=0.17",
                 "safetensors", "sentencepiece", "huggingface_hub", "pillow", "numpy")
    .env({"HF_HOME": "/hf-cache"})
)


@app.cls(image=klein_image, gpu="L40S", volumes={"/hf-cache": klein_vol}, timeout=900, scaledown_window=60, max_containers=4)
class KleinWorker:
    @modal.enter()
    def load(self):
        import torch
        from diffusers import Flux2KleinPipeline
        self.torch = torch
        self.pipe = Flux2KleinPipeline.from_pretrained("black-forest-labs/FLUX.2-klein-4B", torch_dtype=torch.bfloat16).to("cuda")
        self.pipe.load_lora_weights("Alissonerdx/BFS-Best-Face-Swap", weight_name="bfs_head_v1_flux-klein_4b.safetensors", adapter_name="bfs")
        klein_vol.commit()

    @modal.method()
    def process(self, batch: list, head_png: bytes, seed: int = 42, steps: int = 4, prompt: str = "", lora: float = 1.0, res: int = 1024):
        import io
        from PIL import Image
        self.pipe.set_adapters(["bfs"], adapter_weights=[float(lora)])
        head = Image.open(io.BytesIO(head_png)).convert("RGB")
        out = []
        for idx, b in batch:
            body = Image.open(io.BytesIO(b)).convert("RGB").resize((res, res), Image.LANCZOS)
            img = self.pipe(image=[body, head], prompt=prompt or KLEIN_PROMPT, width=res, height=res, num_inference_steps=int(steps),
                            guidance_scale=1.0, generator=self.torch.Generator("cuda").manual_seed(int(seed))).images[0]
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            out.append((idx, buf.getvalue()))
        return out


HARMONIZE_PROMPT = ("Keep this photo exactly the same: same person, same face, same pose, expression, hair, clothes and background. "
                    "Only make the head look naturally part of the photo: consistent lighting and shadows, the same skin tone and "
                    "skin texture on face and neck, a natural jawline and hairline, the same camera focus, sharpness and grain. Photorealistic.")


def _refine_graph(img, seed, steps, denoise, res, prompt=HARMONIZE_PROMPT):
    """low-strength image-to-image with the base Qwen Image 2.1 model (no head-swap LoRA): re-draws only fine detail,
    light and texture so a pasted head blends into the photo, while the content (identity) stays."""
    return {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "qwen_image_2.1_int8_convrot.safetensors", "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen3vl_8b_int8_convrot.safetensors", "type": "qwen_image", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": "qwen_image_2.1_vae_bf16.safetensors"}},
        "4": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["1", 0], "lora_name": "p_q_image_2.1_8_steps_v0.1_comfyui_bf16.safetensors", "strength_model": 1.0}},
        "6": {"class_type": "QwenImage21Cache", "inputs": {"model": ["4", 0], "device": "auto", "dtype": "default"}},
        "7": {"class_type": "LoadImage", "inputs": {"image": img}},
        "9": {"class_type": "ImageScale", "inputs": {"image": ["7", 0], "upscale_method": "lanczos", "width": res, "height": res, "crop": "disabled"}},
        "11": {"class_type": "TextEncodeQwenImage21", "inputs": {"clip": ["2", 0], "prompt": prompt, "negative_prompt": "", "resolution": 0,
                                                                  "images.image_1": ["9", 0], "vae": ["3", 0]}},
        "12": {"class_type": "VAEEncode", "inputs": {"pixels": ["9", 0], "vae": ["3", 0]}},
        "13": {"class_type": "KSampler", "inputs": {"model": ["6", 0], "seed": seed, "steps": steps, "cfg": 1.0, "sampler_name": "euler",
                                                      "scheduler": "simple", "positive": ["11", 0], "negative": ["11", 1], "latent_image": ["12", 0],
                                                      "denoise": denoise}},
        "14": {"class_type": "VAEDecode", "inputs": {"samples": ["13", 0], "vae": ["3", 0]}},
        "15": {"class_type": "SaveImage", "inputs": {"images": ["14", 0], "filename_prefix": "ref"}},
    }


_SNAP_OPTS = dict(enable_memory_snapshot=True, experimental_options={"enable_gpu_snapshot": True}) if USE_SNAPSHOT else {}


# no min_containers: the GPU scales to zero 15 s after the last job, so an idle app costs nothing
@app.cls(image=gpu_image, gpu="L40S", volumes={MODELS: models_vol}, timeout=1800, scaledown_window=15, max_containers=GPU_PARALLEL, **_SNAP_OPTS)
class Worker:
    @modal.enter(snap=USE_SNAPSHOT)
    def boot(self):
        for sub in ["diffusion_models", "text_encoders", "vae", "loras"]:
            d = f"/comfy/models/{sub}"
            subprocess.run(["rm", "-rf", d])
            os.symlink(f"{MODELS}/{sub}", d)
        cmd = ["python", "main.py", "--listen", "127.0.0.1", "--port", "8188"] + (["--use-sage-attention"] if USE_SAGE else [])
        self.proc = subprocess.Popen(cmd, cwd="/comfy", stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for _ in range(300):
            try:
                urllib.request.urlopen("http://127.0.0.1:8188/system_stats", timeout=2)
                break
            except Exception:
                time.sleep(2)
        else:
            raise RuntimeError("comfy did not start")
        if USE_SNAPSHOT:
            self._warmup()   # load every model onto the GPU before the snapshot is taken

    def _warmup(self):
        """One tiny 1-step generation so all weights are loaded (captured by the snapshot)."""
        from PIL import Image
        os.makedirs("/comfy/input", exist_ok=True)
        Image.new("RGB", (256, 256), (128, 128, 128)).save("/comfy/input/warm.png")
        pid = self._post({"prompt": _graph("warm.png", "warm.png", 1, 1, "euler", 256, 256)})["prompt_id"]
        for _ in range(600):
            h = json.load(urllib.request.urlopen(f"http://127.0.0.1:8188/history/{pid}"))
            if pid in h:
                break
            time.sleep(0.5)

    def _post(self, data):
        req = urllib.request.Request("http://127.0.0.1:8188/prompt", data=json.dumps(data).encode(), headers={"Content-Type": "application/json"})
        try:
            return json.load(urllib.request.urlopen(req))
        except urllib.error.HTTPError as e:
            raise RuntimeError(e.read().decode())

    @modal.method()
    def process_seq(self, chain: list, head_png: bytes, steps: int = 8, sampler: str = "euler", seed: int = 42, res: int = BODY_RES, href: int = HEAD_RES):
        """chain: [(idx, crop_png)] in time order. Each frame gets the previous generated frame as a 3rd reference."""
        os.makedirs("/comfy/input", exist_ok=True)
        hname = f"head_{uuid.uuid4().hex[:8]}.png"
        open(f"/comfy/input/{hname}", "wb").write(head_png)
        results, prev_png, t0 = [], None, time.time()
        for idx, b in chain:
            bname = f"b_{uuid.uuid4().hex[:8]}.png"
            open(f"/comfy/input/{bname}", "wb").write(b)
            pname = None
            if prev_png is not None:
                pname = f"p_{uuid.uuid4().hex[:8]}.png"
                open(f"/comfy/input/{pname}", "wb").write(prev_png)
            pid = self._post({"prompt": _graph(bname, hname, seed, steps, sampler, res, href, pname)})["prompt_id"]
            while True:
                h = json.load(urllib.request.urlopen(f"http://127.0.0.1:8188/history/{pid}"))
                if pid in h:
                    break
                time.sleep(0.5)
            if h[pid]["status"].get("status_str") != "success":
                raise RuntimeError("comfy failed: " + json.dumps(h[pid]["status"])[:800])
            im = h[pid]["outputs"]["15"]["images"][0]
            q = urllib.parse.urlencode({"filename": im["filename"], "subfolder": im["subfolder"], "type": im["type"]})
            prev_png = urllib.request.urlopen(f"http://127.0.0.1:8188/view?{q}").read()
            results.append((idx, prev_png))
            os.remove(f"/comfy/input/{bname}")
            if pname:
                os.remove(f"/comfy/input/{pname}")
        print(f"chain of {len(chain)} done in {time.time()-t0:.1f}s", flush=True)
        return results

    @modal.method()
    def process(self, batch: list, head_png: bytes, steps: int = 8, sampler: str = "euler", seed: int = 42, res: int = BODY_RES, href: int = HEAD_RES,
                vae_up: bool = False, out_res: int = 0, cfg: float = 1.0, distill: bool = True, engine: str = "bfs",
                prompt: str = "", lora: float = 1.0):
        """batch: [(idx, crop_png_bytes)] -> [(idx, out_png_bytes)]. engine 'bfs' = Qwen Image 2.1 + BFS head v1.1;
        engine 'edit' = Qwen-Image-Edit-2511 + BFS head V5 (instruction editing, prompt-controlled)"""
        os.makedirs("/comfy/input", exist_ok=True)
        hname = f"head_{uuid.uuid4().hex[:8]}.png"
        open(f"/comfy/input/{hname}", "wb").write(head_png)
        results = []
        t0 = time.time()
        for idx, b in batch:
            bname = f"b_{uuid.uuid4().hex[:8]}.png"
            open(f"/comfy/input/{bname}", "wb").write(b)
            if engine == "edit":
                gr = _edit_graph(bname, hname, seed, steps, res, href, prompt or (EDIT_PROMPT + EDIT_PROMPT_EXTRA),
                                 lightning=distill, lora=lora, cfg=cfg)
            else:
                gr = _graph(bname, hname, seed, steps, sampler, res, href, vae_up=vae_up, out_res=out_res, cfg=cfg, distill=distill)
            pid = self._post({"prompt": gr})["prompt_id"]
            while True:
                h = json.load(urllib.request.urlopen(f"http://127.0.0.1:8188/history/{pid}"))
                if pid in h:
                    break
                time.sleep(0.5)
            if h[pid]["status"].get("status_str") != "success":
                raise RuntimeError("comfy failed: " + json.dumps(h[pid]["status"])[:800])
            im = next(o for o in h[pid]["outputs"].values() if o.get("images"))["images"][0]
            q = urllib.parse.urlencode({"filename": im["filename"], "subfolder": im["subfolder"], "type": im["type"]})
            results.append((idx, urllib.request.urlopen(f"http://127.0.0.1:8188/view?{q}").read()))
            os.remove(f"/comfy/input/{bname}")
        print(f"batch of {len(batch)} done in {time.time()-t0:.1f}s", flush=True)
        return results

    @modal.method()
    def refine(self, img_png: bytes, seed: int = 42, steps: int = 8, denoise: float = 0.3, res: int = BODY_RES, prompt: str = "") -> bytes:
        """harmonize pass: low-strength image-to-image of a finished crop (see _refine_graph); with a prompt and
        denoise 1.0 it is an instruction edit of the image (e.g. remove glasses from a reference photo)"""
        os.makedirs("/comfy/input", exist_ok=True)
        name = f"r_{uuid.uuid4().hex[:8]}.png"
        open(f"/comfy/input/{name}", "wb").write(img_png)
        pid = self._post({"prompt": _refine_graph(name, seed, steps, denoise, res, prompt or HARMONIZE_PROMPT)})["prompt_id"]
        while True:
            h = json.load(urllib.request.urlopen(f"http://127.0.0.1:8188/history/{pid}"))
            if pid in h:
                break
            time.sleep(0.5)
        if h[pid]["status"].get("status_str") != "success":
            raise RuntimeError("comfy failed: " + json.dumps(h[pid]["status"])[:800])
        im = h[pid]["outputs"]["15"]["images"][0]
        q = urllib.parse.urlencode({"filename": im["filename"], "subfolder": im["subfolder"], "type": im["type"]})
        os.remove(f"/comfy/input/{name}")
        return urllib.request.urlopen(f"http://127.0.0.1:8188/view?{q}").read()


# ----------------------------------------------------------------------------- temporal refiner (Wan 2.2 TI2V-5B, Apache-2.0)
# BFS generates every frame on its own, so fine hair / skin detail shimmers. The refiner re-draws the swapped head
# clip with a VIDEO model at low strength (video-to-video): the model sees up to 81 frames at once and makes them
# consistent, while the low strength keeps the BFS identity.
WAN_PROMPT = ("a realistic video of a young man with short dark hair, natural skin texture, detailed hair, the same face "
              "in every frame, steady, sharp focus, natural lighting")
WAN_NEG = "flicker, jitter, morphing, blurry, distorted face, deformed, artifacts, low quality, cartoon, watermark"


def _wan_graph(video_name, seed, steps, cfg, denoise, prompt=WAN_PROMPT, neg=WAN_NEG):
    return {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "wan2.2_ti2v_5B_fp16.safetensors", "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": "umt5_xxl_fp8_e4m3fn_scaled.safetensors", "type": "wan", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": "wan2.2_vae.safetensors"}},
        "4": {"class_type": "ModelSamplingSD3", "inputs": {"model": ["1", 0], "shift": 8.0}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": prompt}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": neg}},
        "7": {"class_type": "LoadVideo", "inputs": {"file": video_name}},
        "8": {"class_type": "GetVideoComponents", "inputs": {"video": ["7", 0]}},
        "9": {"class_type": "VAEEncode", "inputs": {"pixels": ["8", 0], "vae": ["3", 0]}},
        "10": {"class_type": "KSampler", "inputs": {"model": ["4", 0], "seed": seed, "steps": steps, "cfg": cfg, "sampler_name": "uni_pc",
                                                      "scheduler": "simple", "positive": ["5", 0], "negative": ["6", 0], "latent_image": ["9", 0],
                                                      "denoise": denoise}},
        "11": {"class_type": "VAEDecode", "inputs": {"samples": ["10", 0], "vae": ["3", 0]}},
        "12": {"class_type": "SaveImage", "inputs": {"images": ["11", 0], "filename_prefix": "wan"}},
    }


@app.cls(image=gpu_image, gpu="L40S", volumes={MODELS: models_vol}, timeout=1800, scaledown_window=15, max_containers=3)
class RefineWorker:
    @modal.enter()
    def boot(self):
        for sub in ["diffusion_models", "text_encoders", "vae", "loras"]:
            d = f"/comfy/models/{sub}"
            subprocess.run(["rm", "-rf", d])
            os.symlink(f"{MODELS}/{sub}", d)
        self.proc = subprocess.Popen(["python", "main.py", "--listen", "127.0.0.1", "--port", "8188"], cwd="/comfy",
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for _ in range(300):
            try:
                urllib.request.urlopen("http://127.0.0.1:8188/system_stats", timeout=2)
                break
            except Exception:
                time.sleep(2)
        else:
            raise RuntimeError("comfy did not start")

    @modal.method()
    def refine(self, clip_mp4: bytes, denoise: float = 0.3, steps: int = 20, cfg: float = 5.0, seed: int = 42,
               prompt: str = WAN_PROMPT) -> list:
        """clip_mp4: head clip (4k+1 frames, side a multiple of 32) -> list of refined PNG frames (same size/order)"""
        os.makedirs("/comfy/input", exist_ok=True)
        name = f"clip_{uuid.uuid4().hex[:8]}.mp4"
        open(f"/comfy/input/{name}", "wb").write(clip_mp4)
        t0 = time.time()
        req = urllib.request.Request("http://127.0.0.1:8188/prompt", headers={"Content-Type": "application/json"},
                                     data=json.dumps({"prompt": _wan_graph(name, seed, steps, cfg, denoise, prompt)}).encode())
        try:
            pid = json.load(urllib.request.urlopen(req))["prompt_id"]
        except urllib.error.HTTPError as e:
            raise RuntimeError(e.read().decode()[:1500])
        while True:
            h = json.load(urllib.request.urlopen(f"http://127.0.0.1:8188/history/{pid}"))
            if pid in h:
                break
            time.sleep(1)
        if h[pid]["status"].get("status_str") != "success":
            raise RuntimeError("comfy failed: " + json.dumps(h[pid]["status"])[:1500])
        out = []
        for im in h[pid]["outputs"]["12"]["images"]:
            q = urllib.parse.urlencode({"filename": im["filename"], "subfolder": im["subfolder"], "type": im["type"]})
            out.append(urllib.request.urlopen(f"http://127.0.0.1:8188/view?{q}").read())
        os.remove(f"/comfy/input/{name}")
        print(f"refined clip of {len(out)} frames in {time.time()-t0:.1f}s", flush=True)
        return out


# ----------------------------------------------------------------------------- video-native head replacement (Wan 2.2 Animate 14B)
# The head is generated as ONE video (77/81 frames at once) instead of frame by frame, so there is no per-frame
# randomness to cause jitter. Inputs (all in head-crop space): reference = one BFS-swapped frame (identity, hair, eye
# colour), face_video = the original performer's face crops (expressions), pose_video = DWPose of the original crop
# (head/shoulder pose), background_video = original crop with the head blacked out + character_mask = head mask.
anim_image = gpu_image.run_commands(
    "cd /comfy/custom_nodes && git clone --depth 1 https://github.com/Fannovel16/comfyui_controlnet_aux",
    "cd /comfy/custom_nodes/comfyui_controlnet_aux && pip install -r requirements.txt",
)
ANIM_NEG = ("ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¨ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â°ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â²ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¨ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â°ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¨ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â°ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â³ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¤ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¸ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â½ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¯ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¼ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¨ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¿ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¦ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚ÂºÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚ÂÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¯ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¼ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â©ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚ÂÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¾ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¦ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚ÂÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¯ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¼ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â§ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â»ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¨ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¦ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¨ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â§ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â³ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¤ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¸ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚ÂÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¦ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¸ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¦ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¯ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¼ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¥ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â­ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚ÂÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¥ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¹ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¯ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¼ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â©ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â£ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â½ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¦ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¼ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¯ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¼ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¤ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â½ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã¢â‚¬Å“ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¥ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦ÃƒÂ¢Ã¢â€šÂ¬Ã…â€œÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚ÂÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¯ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¼ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â§ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚ÂÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â»ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¤ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â½ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã¢â‚¬Å“ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¯ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¼ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â§ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚ÂÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â»ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â©ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚ÂÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¯ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¼ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â©ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚ÂÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¾ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¦ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â­ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¯ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¼ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¦ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â´ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¤ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â½ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦ÃƒÂ¢Ã¢â€šÂ¬Ã…â€œÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¥ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚ÂÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¹Ãƒâ€¦Ã¢â‚¬Å“ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â§ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚ÂÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â°ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¯ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¼ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¦ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã¢â‚¬Å“ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¥ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â·ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â®ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¨ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â´ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¨ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â©ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚ÂÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¯ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¼ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¤ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â½ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â½ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¨ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â´ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¨ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â©ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚ÂÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¯ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¼ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢JPEGÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¥ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â½ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¹ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â§ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¼ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â©ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¦ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â®ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¹ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â§ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¾ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¯ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¼ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¤ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¸ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¹Ãƒâ€¦Ã¢â‚¬Å“ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â©ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¾ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¹ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â§ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¾ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¯ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¼ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¦ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â®ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¹ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â§ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¼ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚ÂºÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â§ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¾ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¯ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¼ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢"
            "ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¥ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¤ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¤ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â½ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¾ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â§ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¾ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¦ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â°ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¹ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¦ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¯ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¼ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â§ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚ÂÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â»ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¥ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¾ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚ÂÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¤ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¸ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚ÂÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¥ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¥ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â½ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â§ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¾ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¦ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â°ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¹ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â©ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¨ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¯ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¼ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â§ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚ÂÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â»ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¥ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¾ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚ÂÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¤ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¸ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚ÂÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¥ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¥ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â½ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â§ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¾ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¨ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¾ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¸ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â©ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¨ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¯ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¼ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â§ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¸ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¥ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â½ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â§ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¾ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¯ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¼ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¦ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¯ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚ÂÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¥ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â®ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¹ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â§ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¾ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¯ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¼ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¥ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â½ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¦ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚ÂÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â§ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¸ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¥ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â½ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â§ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¾ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¨ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¤ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â½ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦ÃƒÂ¢Ã¢â€šÂ¬Ã…â€œÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¯ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¼ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¦ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â°ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¹ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¦ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¨ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¾ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚ÂÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¥ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚ÂÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¹ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¯ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¼ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â©ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚ÂÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¾ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¦ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â­ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¤ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¸ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚ÂÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¥ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¨ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â§ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¾ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â§ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚ÂÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â»ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â©ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚ÂÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¯ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¼ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¦ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚ÂÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¤ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¹ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â±ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â§ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¾ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¨ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¦ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¾ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¯")
ANIM_PROMPT = ("the man from the reference image, realistic video, natural skin texture, detailed hair, the same face in every "
               "frame, natural lighting")


def _anim_graph(w, h, length, seed, steps, prompt):
    return {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "wan2.2_animate_14B_int8_convrot.safetensors", "weight_dtype": "default"}},
        "2": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["1", 0], "lora_name": "lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors", "strength_model": 1.0}},
        "3": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["2", 0], "lora_name": "wan2.2_animate_14B_relight_lora_bf16.safetensors", "strength_model": 1.0}},
        "4": {"class_type": "ModelSamplingSD3", "inputs": {"model": ["3", 0], "shift": 8.0}},
        "5": {"class_type": "CLIPLoader", "inputs": {"clip_name": "umt5_xxl_fp8_e4m3fn_scaled.safetensors", "type": "wan", "device": "default"}},
        "6": {"class_type": "VAELoader", "inputs": {"vae_name": "wan_2.1_vae.safetensors"}},
        "7": {"class_type": "CLIPVisionLoader", "inputs": {"clip_name": "clip_vision_h.safetensors"}},
        "8": {"class_type": "LoadImage", "inputs": {"image": "ref.png"}},
        "9": {"class_type": "CLIPVisionEncode", "inputs": {"clip_vision": ["7", 0], "image": ["8", 0], "crop": "none"}},
        "10": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["5", 0], "text": prompt}},
        "11": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["5", 0], "text": ANIM_NEG}},
        "12": {"class_type": "LoadVideo", "inputs": {"file": "bg.mp4"}},
        "13": {"class_type": "GetVideoComponents", "inputs": {"video": ["12", 0]}},
        "14": {"class_type": "LoadVideo", "inputs": {"file": "face.mp4"}},
        "15": {"class_type": "GetVideoComponents", "inputs": {"video": ["14", 0]}},
        "16": {"class_type": "LoadVideo", "inputs": {"file": "orig.mp4"}},
        "17": {"class_type": "GetVideoComponents", "inputs": {"video": ["16", 0]}},
        "18": {"class_type": "DWPreprocessor", "inputs": {"image": ["17", 0], "detect_hand": "enable", "detect_body": "enable", "detect_face": "disable",
                                                          "resolution": 512, "bbox_detector": "yolox_l.onnx",
                                                          "pose_estimator": "dw-ll_ucoco_384_bs5.torchscript.pt", "scale_stick_for_xinsr_cn": "disable"}},
        "19": {"class_type": "LoadVideo", "inputs": {"file": "mask.mp4"}},
        "20": {"class_type": "GetVideoComponents", "inputs": {"video": ["19", 0]}},
        "21": {"class_type": "ImageToMask", "inputs": {"image": ["20", 0], "channel": "red"}},
        "22": {"class_type": "WanAnimateToVideo", "inputs": {"positive": ["10", 0], "negative": ["11", 0], "vae": ["6", 0], "width": w, "height": h,
                                                             "length": length, "batch_size": 1, "clip_vision_output": ["9", 0], "reference_image": ["8", 0],
                                                             "face_video": ["15", 0], "pose_video": ["18", 0], "background_video": ["13", 0],
                                                             "character_mask": ["21", 0], "continue_motion_max_frames": 5, "video_frame_offset": 0}},
        "23": {"class_type": "KSampler", "inputs": {"model": ["4", 0], "seed": seed, "steps": steps, "cfg": 1.0, "sampler_name": "euler",
                                                      "scheduler": "simple", "positive": ["22", 0], "negative": ["22", 1], "latent_image": ["22", 2],
                                                      "denoise": 1.0}},
        "24": {"class_type": "TrimVideoLatent", "inputs": {"samples": ["23", 0], "trim_amount": ["22", 3]}},
        "25": {"class_type": "VAEDecode", "inputs": {"samples": ["24", 0], "vae": ["6", 0]}},
        "26": {"class_type": "SaveImage", "inputs": {"images": ["25", 0], "filename_prefix": "anim"}},
    }


@app.cls(image=anim_image, gpu="L40S", volumes={MODELS: models_vol}, timeout=3600, scaledown_window=15, max_containers=3)
class AnimateWorker:
    @modal.enter()
    def boot(self):
        for sub in ["diffusion_models", "text_encoders", "vae", "loras", "clip_vision"]:
            d = f"/comfy/models/{sub}"
            subprocess.run(["rm", "-rf", d])
            os.symlink(f"{MODELS}/{sub}", d)
        self.proc = subprocess.Popen(["python", "main.py", "--listen", "127.0.0.1", "--port", "8188"], cwd="/comfy",
                                     stdout=open("/tmp/comfy.log", "w"), stderr=subprocess.STDOUT)
        for _ in range(450):
            try:
                urllib.request.urlopen("http://127.0.0.1:8188/system_stats", timeout=2)
                break
            except Exception:
                time.sleep(2)
        else:
            raise RuntimeError("comfy did not start: " + open("/tmp/comfy.log").read()[-3000:])

    @modal.method()
    def animate(self, files: dict, w: int, h: int, length: int, seed: int = 42, steps: int = 6, prompt: str = ANIM_PROMPT) -> list:
        """files: {'ref.png','bg.mp4','face.mp4','orig.mp4','mask.mp4'} -> list of PNG frames (length)"""
        os.makedirs("/comfy/input", exist_ok=True)
        for k, v in files.items():
            open(f"/comfy/input/{k}", "wb").write(v)
        t0 = time.time()
        req = urllib.request.Request("http://127.0.0.1:8188/prompt", headers={"Content-Type": "application/json"},
                                     data=json.dumps({"prompt": _anim_graph(w, h, length, seed, steps, prompt)}).encode())
        try:
            pid = json.load(urllib.request.urlopen(req))["prompt_id"]
        except urllib.error.HTTPError as e:
            raise RuntimeError(e.read().decode()[:3000])
        while True:
            hst = json.load(urllib.request.urlopen(f"http://127.0.0.1:8188/history/{pid}"))
            if pid in hst:
                break
            time.sleep(2)
        if hst[pid]["status"].get("status_str") != "success":
            raise RuntimeError("comfy failed: " + json.dumps(hst[pid]["status"])[:3000] + "\nLOG:" + open("/tmp/comfy.log").read()[-3000:])
        out = []
        for im in hst[pid]["outputs"]["26"]["images"]:
            q = urllib.parse.urlencode({"filename": im["filename"], "subfolder": im["subfolder"], "type": im["type"]})
            out.append(urllib.request.urlopen(f"http://127.0.0.1:8188/view?{q}").read())
        print(f"animated {len(out)} frames in {time.time()-t0:.1f}s", flush=True)
        return out


# ----------------------------------------------------------------------------- user workflow ("qwen edit inpaint")
# Port of the ComfyUI workflow the user supplied: Qwen-Image-Edit-2509 fp8 + Lightning 4-step V2, inpainting
# (InpaintModelConditioning + DifferentialDiffusion, denoise 0.7) inside a grown/blurred head mask, InstantX
# ControlNet Union on a DepthAnythingV2 map (0.9, end 0.8), ClownsharKSampler res_2s / sigmoid_offset 4 steps,
# Wan2.1 2x-upscale VAE decode -> 0.5x, then SeedVR2 7B upscale blended 50/50 with a lanczos 2x upscale.
# Changes for video: Gemini captions replaced by ONE fixed prompt (no external API, no per-frame prompt changes),
# fixed seeds, the person is tracked and processed in a head crop, result pasted back with the same mask.
# Custom nodes pinned to the commits the workflow was saved with.
user_image = anim_image.run_commands(
    "cd /comfy/custom_nodes && git clone https://github.com/ClownsharkBatwing/RES4LYF && cd RES4LYF && git checkout 46de917234f9fef3f2ab411c41e07aa3c633f4f7 && pip install -r requirements.txt",
    "cd /comfy/custom_nodes && git clone https://github.com/spacepxl/ComfyUI-VAE-Utils",
    "cd /comfy/custom_nodes && git clone https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler && cd ComfyUI-SeedVR2_VideoUpscaler && git checkout 984f9af478cbde993c36f1006625753498d2c08d && pip install -r requirements.txt",
    # the unpinned torchaudio in the base image does not match torch 2.8 (undefined symbol) and breaks diffusers,
    # which SeedVR2 imports; nothing here needs audio
    "pip uninstall -y torchaudio",
)
USER_PROMPT = ("A young man eating a burger in a fast-food restaurant, photorealistic, natural indoor lighting. "
               "A man with fair skin, short dark brown textured hair swept up, and light blue-grey eyes.")


def _user_graph(seed, prompt, up_res):
    return {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "qwen_image_edit_2509_fp8_e4m3fn.safetensors", "weight_dtype": "default"}},
        "2": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["1", 0], "lora_name": "Qwen-Image-Lightning-4steps-V2.0.safetensors", "strength_model": 1.0}},
        "3": {"class_type": "DifferentialDiffusion", "inputs": {"model": ["2", 0]}},
        "4": {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": ["3", 0], "shift": 2.0}},
        "5": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen_2.5_vl_7b_fp8_scaled.safetensors", "type": "qwen_image", "device": "default"}},
        "6": {"class_type": "VAELoader", "inputs": {"vae_name": "qwen_image_vae.safetensors"}},
        "7": {"class_type": "VAEUtils_CustomVAELoader", "inputs": {"vae_name": "Wan2.1_VAE_upscale2x_imageonly_real_v1.safetensors", "disable_offload": True, "precision": "auto"}},
        "8": {"class_type": "LoadImage", "inputs": {"image": "target.png"}},
        "9": {"class_type": "LoadImage", "inputs": {"image": "mask.png"}},
        "10": {"class_type": "ImageToMask", "inputs": {"image": ["9", 0], "channel": "red"}},
        "11": {"class_type": "LoadImage", "inputs": {"image": "source.png"}},
        "12": {"class_type": "TextEncodeQwenImageEditPlus", "inputs": {"clip": ["5", 0], "prompt": prompt, "vae": ["6", 0], "image1": ["11", 0]}},
        "13": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["12", 0]}},
        "14": {"class_type": "InpaintModelConditioning", "inputs": {"positive": ["12", 0], "negative": ["13", 0], "vae": ["6", 0],
                                                                    "pixels": ["8", 0], "mask": ["10", 0], "noise_mask": True}},
        "15": {"class_type": "ImageScaleToTotalPixels", "inputs": {"image": ["8", 0], "upscale_method": "nearest-exact", "megapixels": 0.24, "resolution_steps": 1}},
        "16": {"class_type": "DepthAnythingV2Preprocessor", "inputs": {"image": ["15", 0], "ckpt_name": "depth_anything_v2_vitl.pth", "resolution": 512}},
        "17": {"class_type": "ControlNetLoader", "inputs": {"control_net_name": "Qwen-Image-InstantX-ControlNet-Union.safetensors"}},
        "18": {"class_type": "ControlNetApplyAdvanced", "inputs": {"positive": ["14", 0], "negative": ["14", 1], "control_net": ["17", 0],
                                                                   "image": ["16", 0], "vae": ["6", 0], "strength": 0.9, "start_percent": 0.0, "end_percent": 0.8}},
        "19": {"class_type": "ClownsharKSampler_Beta", "inputs": {"model": ["4", 0], "positive": ["18", 0], "negative": ["18", 1], "latent_image": ["14", 2],
                                                                  "eta": 0.5, "sampler_name": "exponential/res_2s", "scheduler": "bong_tangent", "steps": 4,
                                                                  "steps_to_run": -1, "denoise": 0.7, "cfg": 1.0, "seed": seed, "sampler_mode": "standard",
                                                                  "bongmath": False}},
        "20": {"class_type": "VAEUtils_VAEDecodeTiled", "inputs": {"samples": ["19", 0], "vae": ["7", 0], "upscale": -1, "tile": False, "tile_size": 512,
                                                                   "overlap": 64, "temporal_size": 4096, "temporal_overlap": 64}},
        "21": {"class_type": "ImageScaleBy", "inputs": {"image": ["20", 0], "upscale_method": "lanczos", "scale_by": 0.5}},
        "22": {"class_type": "SeedVR2ExtraArgs", "inputs": {"tiled_vae": True, "vae_tile_size": 928, "vae_tile_overlap": 256, "preserve_vram": True,
                                                            "cache_model": True, "enable_debug": False, "device": "cuda:0"}},
        "23": {"class_type": "SeedVR2", "inputs": {"images": ["21", 0], "model": "seedvr2_ema_7b_fp16.safetensors", "seed": seed, "new_resolution": up_res,
                                                   "batch_size": 5, "color_correction": "wavelet", "input_noise_scale": 0.0, "latent_noise_scale": 0.0,
                                                   "extra_args": ["22", 0]}},
        "24": {"class_type": "ImageScaleBy", "inputs": {"image": ["21", 0], "upscale_method": "lanczos", "scale_by": 2.0}},
        "25": {"class_type": "ImageBlend", "inputs": {"image1": ["24", 0], "image2": ["23", 0], "blend_factor": 0.5, "blend_mode": "normal"}},
        "26": {"class_type": "SaveImage", "inputs": {"images": ["25", 0], "filename_prefix": "uf"}},
    }


@app.cls(image=user_image, gpu="L40S", volumes={MODELS: models_vol}, timeout=3600, scaledown_window=15, max_containers=GPU_PARALLEL)
class UserFlowWorker:
    @modal.enter()
    def boot(self):
        for sub in ["diffusion_models", "text_encoders", "vae", "loras", "controlnet", "SEEDVR2"]:
            d = f"/comfy/models/{sub}"
            subprocess.run(["rm", "-rf", d])
            os.symlink(f"{MODELS}/{sub}", d)
        self.proc = subprocess.Popen(["python", "main.py", "--listen", "127.0.0.1", "--port", "8188"], cwd="/comfy",
                                     stdout=open("/tmp/comfy.log", "w"), stderr=subprocess.STDOUT)
        for _ in range(450):
            try:
                urllib.request.urlopen("http://127.0.0.1:8188/system_stats", timeout=2)
                break
            except Exception:
                time.sleep(2)
        else:
            raise RuntimeError("comfy did not start: " + open("/tmp/comfy.log").read()[-3000:])

    @modal.method()
    def process(self, batch: list, source_png: bytes, seed: int = 42, prompt: str = USER_PROMPT, up_res: int = 2560) -> list:
        """batch: [(idx, target_png, mask_png)] -> [(idx, out_png)]"""
        os.makedirs("/comfy/input", exist_ok=True)
        open("/comfy/input/source.png", "wb").write(source_png)
        out, t0 = [], time.time()
        for idx, tp, mp in batch:
            open("/comfy/input/target.png", "wb").write(tp)
            open("/comfy/input/mask.png", "wb").write(mp)
            req = urllib.request.Request("http://127.0.0.1:8188/prompt", headers={"Content-Type": "application/json"},
                                         data=json.dumps({"prompt": _user_graph(seed, prompt, up_res)}).encode())
            try:
                pid = json.load(urllib.request.urlopen(req))["prompt_id"]
            except urllib.error.HTTPError as e:
                lg = open("/tmp/comfy.log").read()
                bad = [ln for ln in lg.splitlines() if "IMPORT FAILED" in ln or "Error" in ln or "No module" in ln]
                raise RuntimeError(e.read().decode()[:6000] + " IMPORT LOG: " + " | ".join(bad[-30:])[:4000])
            while True:
                hst = json.load(urllib.request.urlopen(f"http://127.0.0.1:8188/history/{pid}"))
                if pid in hst:
                    break
                time.sleep(0.5)
            if hst[pid]["status"].get("status_str") != "success":
                raise RuntimeError("comfy failed: " + json.dumps(hst[pid]["status"])[:3000] + "\nLOG:" + open("/tmp/comfy.log").read()[-3000:])
            im = hst[pid]["outputs"]["26"]["images"][0]
            q = urllib.parse.urlencode({"filename": im["filename"], "subfolder": im["subfolder"], "type": im["type"]})
            out.append((idx, urllib.request.urlopen(f"http://127.0.0.1:8188/view?{q}").read()))
        print(f"userflow batch of {len(batch)} done in {time.time()-t0:.1f}s", flush=True)
        return out


# ----------------------------------------------------------------------------- CPU helpers
def _fa():
    from insightface.app import FaceAnalysis
    fa = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"], allowed_modules=["detection", "recognition"])
    fa.prepare(ctx_id=-1, det_size=(640, 640))
    return fa


_FA68 = {}


def _fa68():
    """detector + 68-point 3D landmarks (gives head pose pitch/yaw/roll and expression points)"""
    if "m" not in _FA68:
        from insightface.app import FaceAnalysis
        m = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"], allowed_modules=["detection", "landmark_3d_68"])
        m.prepare(ctx_id=-1, det_size=(640, 640))
        _FA68["m"] = m
    return _FA68["m"]


def _expr_pose_diff(f_a, f_b):
    """expression difference (mouth + eyes shape after aligning the faces, in inter-ocular units) and pose difference (deg)"""
    import numpy as np
    pa, pb = f_a.landmark_3d_68[:, :2].astype(np.float64), f_b.landmark_3d_68[:, :2].astype(np.float64)
    stable = [27, 28, 29, 30, 36, 39, 42, 45]          # nose bridge + eye corners
    A, B = pb[stable], pa[stable]
    ma, mb = A.mean(0), B.mean(0)
    A0, B0 = A - ma, B - mb
    U, S, Vt = np.linalg.svd(A0.T @ B0)
    R = (U @ Vt).T
    s = S.sum() / (A0 ** 2).sum()
    pb_al = (pb - ma) @ R.T * s + mb                    # face b aligned onto face a
    iod = float(np.linalg.norm(pa[36] - pa[45])) + 1e-6
    mouth = float(np.linalg.norm(pb_al[48:68] - pa[48:68], axis=1).mean() / iod)
    eyes = float(np.linalg.norm(pb_al[36:48] - pa[36:48], axis=1).mean() / iod)
    pose = float(np.abs(np.asarray(f_a.pose) - np.asarray(f_b.pose)).sum()) if getattr(f_a, "pose", None) is not None else 0.0
    return mouth + 0.5 * eyes, pose


def _relight(gen, orig, lo, lg, face_w, alpha=0.45):
    """carry over the broad light/shadow pattern of the original face skin onto the new face skin (low frequencies only)"""
    import cv2, numpy as np
    sk = [1, 2, 17]
    mo = np.isin(lo, sk).astype(np.float32)
    mg = np.isin(lg, sk).astype(np.float32)
    if mo.sum() < 500 or mg.sum() < 500:
        return gen
    sig = max(3.0, 0.18 * face_w)
    Lo = cv2.cvtColor(orig, cv2.COLOR_BGR2LAB).astype(np.float32)
    G = cv2.cvtColor(gen, cv2.COLOR_BGR2LAB).astype(np.float32)
    shade_o = cv2.GaussianBlur(Lo[..., 0] * mo, (0, 0), sig) / (cv2.GaussianBlur(mo, (0, 0), sig) + 1e-3)
    shade_g = cv2.GaussianBlur(G[..., 0] * mg, (0, 0), sig) / (cv2.GaussianBlur(mg, (0, 0), sig) + 1e-3)
    valid = cv2.GaussianBlur(mo, (0, 0), sig) > 0.25
    d = (shade_o - shade_g) - float(np.median((shade_o - shade_g)[valid & (mg > 0)])) if (valid & (mg > 0)).any() else 0 * shade_o
    w = cv2.GaussianBlur(mg, (0, 0), 0.01 * gen.shape[0] + 1) * valid.astype(np.float32)
    w = w * _not_beard(G[..., 0], lg)      # not on beard / stubble
    G[..., 0] = G[..., 0] + alpha * np.clip(d, -40, 40) * w
    return cv2.cvtColor(np.clip(G, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)


def _color_skin(im):
    import cv2
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    L = cv2.cvtColor(im, cv2.COLOR_BGR2LAB)
    return (hsv[..., 0] <= 22) & (hsv[..., 1] > 35) & (hsv[..., 1] < 200) & (L[..., 0] > 60) & (L[..., 0] < 240)


def _finish_tone(orig, final, mw, box):
    """Final skin-tone pass on the full image: the new FACE skin is matched to the original model's face skin and the new
    NECK skin to the original neck skin (two zones, smooth blend under the jaw). Only skin pixels inside the pasted area
    change - hair, brows, beard, eyes, lips, clothes and everything outside stay exactly as they are."""
    import cv2, numpy as np
    x0, y0, x1, y1 = [int(v) for v in box]
    h, w = y1 - y0, x1 - x0
    H, W = orig.shape[:2]
    R = mw > 0.05
    Ln = cv2.cvtColor(final, cv2.COLOR_BGR2LAB).astype(np.float32)
    Lo = cv2.cvtColor(orig, cv2.COLOR_BGR2LAB).astype(np.float32)
    sn, so = _color_skin(final) & R, _color_skin(orig)

    def zone(a, b, c0, c1):
        z = np.zeros((H, W), bool)
        z[max(0, int(a)):min(H, int(b)), max(0, int(c0)):min(W, int(c1))] = True
        return z

    fz = zone(y0 + .25 * h, y0 + .70 * h, x0 + .2 * w, x1 - .2 * w)            # cheeks / nose
    nz = zone(y1 + .15 * h, y1 + .55 * h, x0 + .25 * w, x1 - .25 * w)          # neck

    def delta(z):
        a, b = z & sn, z & so
        if a.sum() < 150 or b.sum() < 150:
            return None
        d = np.median(Lo[b], 0) - np.median(Ln[a], 0)
        d[0] = np.clip(d[0], -35, 35)
        d[1:] = np.clip(d[1:], -10, 10)
        return d

    df, dn = delta(fz), delta(nz)
    if df is None:
        return final, {}
    if dn is None:
        dn = df
    yy = np.arange(H, dtype=np.float32)[:, None, None]
    ya, yb = y1 - .05 * h, y1 + .2 * h
    t = np.clip((yy - ya) / max(1.0, yb - ya), 0, 1)
    D = df[None, None, :] * (1 - t) + dn[None, None, :] * t
    med = float(np.median(Ln[fz & sn][:, 0])) if (fz & sn).sum() > 100 else 60.0
    wL = np.clip((Ln[..., 0] - (med - 35)) / 15, 0, 1)                         # no shift on beard / brows / hair
    Wt = np.clip(cv2.GaussianBlur(sn.astype(np.float32), (0, 0), max(3, 0.08 * h)) * 1.8, 0, 1) * cv2.GaussianBlur(wL, (0, 0), 3)
    Wt *= cv2.GaussianBlur(R.astype(np.float32), (0, 0), 5)
    out = cv2.cvtColor(np.clip(Ln + D * Wt[..., None], 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
    out = np.where((Wt > 0.003)[..., None], out, final)
    return out, {"face_shift_L": round(float(df[0]), 1), "neck_shift_L": round(float(dn[0]), 1)}


def _changed_region(orig, final):
    """bit-exact changed area (head+neck) of a finished swap: largest changed blob, filled"""
    import cv2, numpy as np
    d = np.abs(orig.astype(int) - final.astype(int)).max(2)
    big = cv2.morphologyEx((d > 20).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((41, 41), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(big)
    if n < 2:
        return None, d
    k = 1 + int(np.argmax(st[1:, 4]))
    cnts, _ = cv2.findContours((lab == k).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    R = np.zeros(d.shape, np.uint8)
    cv2.drawContours(R, cnts, -1, 1, -1)
    return R, d


def _qa(orig, final):
    """quality gate numbers: clothes_px (changed pixels far from the head, must be 0), face_tone (new face skin minus the
    original model's face skin, brightness), misjoin (difference of the jaw->neck brightness step vs the original)"""
    import cv2, numpy as np
    R, d = _changed_region(orig, final)
    if R is None:
        return {"clothes_px": 0, "face_tone": 0.0, "misjoin": 0.0, "score": 0.0}
    far = cv2.dilate(R, np.ones((61, 61), np.uint8)) == 0
    ys, xs = np.nonzero(R)
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    h, w = y1 - y0, x1 - x0
    cx = (x0 + x1) // 2
    Ln = cv2.cvtColor(final, cv2.COLOR_BGR2LAB).astype(np.float32)
    Lo = cv2.cvtColor(orig, cv2.COLOR_BGR2LAB).astype(np.float32)
    sn, so = _color_skin(final) & (R > 0), _color_skin(orig)

    def band(a, b, c0, c1):
        z = np.zeros(d.shape, bool)
        z[int(a):int(b), int(c0):int(c1)] = True
        return z

    def med(L, z, s):
        m = z & s
        return float(np.median(L[m][:, 0])) if m.sum() > 80 else None

    face = band(y0 + .28 * h, y0 + .55 * h, cx - .18 * w, cx + .18 * w)
    jaw = band(y0 + .55 * h, y0 + .66 * h, cx - .18 * w, cx + .18 * w)
    nk = band(y0 + .70 * h, y0 + .82 * h, cx - .12 * w, cx + .12 * w)
    fn, fo = med(Ln, face, sn), med(Lo, face, so)
    jn, nn, jo, no = med(Ln, jaw, sn), med(Ln, nk, sn), med(Lo, jaw, so), med(Lo, nk, so)
    q = {"clothes_px": int((d > 0)[far].sum()),
         "face_tone": round(fn - fo, 1) if None not in (fn, fo) else 0.0,
         "misjoin": round(abs((jn - nn) - (jo - no)), 1) if None not in (jn, nn, jo, no) else 0.0}
    q["score"] = round(abs(q["face_tone"]) + q["misjoin"] + (100 if q["clothes_px"] else 0), 1)
    return q


def _profile_fix(orig, final):
    """repair for a face->neck misjoin: match the head/neck skin brightness profile row by row to the original model
    (copies the original's light fall-off and the shadow under the chin). Only skin pixels of the head/neck change."""
    import cv2, numpy as np
    R, d = _changed_region(orig, final)
    if R is None:
        return final
    ys, xs = np.nonzero(R)
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    h, w = y1 - y0, x1 - x0
    cx = (x0 + x1) // 2
    Ln = cv2.cvtColor(final, cv2.COLOR_BGR2LAB).astype(np.float32)
    Lo = cv2.cvtColor(orig, cv2.COLOR_BGR2LAB).astype(np.float32)
    sn, so = _color_skin(final) & (R > 0), _color_skin(orig)
    colz = np.zeros(d.shape, bool)
    colz[:, int(cx - .28 * w):int(cx + .28 * w)] = True
    medf = float(np.median(Ln[sn & colz][:, 0])) if (sn & colz).sum() > 100 else 60.0
    notbeard = Ln[..., 0] > (medf - 30)
    step = max(4, int(.04 * h))
    rows, deltas = [], []
    for y in range(int(y0 + .25 * h), int(min(orig.shape[0] - 1, y1 + .02 * h)), step):
        a = np.zeros(d.shape, bool)
        a[y:y + 2 * step] = True
        a &= colz
        mn, mo = a & sn & notbeard, a & so
        if mn.sum() > 60 and mo.sum() > 60:
            rows.append(y + step)
            deltas.append(np.median(Lo[mo], 0) - np.median(Ln[mn], 0))
    if len(rows) < 3:
        return final
    rows, deltas = np.array(rows, float), np.array(deltas)
    deltas[:, 0] = np.clip(deltas[:, 0], -25, 25)
    deltas[:, 1:] = np.clip(deltas[:, 1:], -8, 8)
    if len(rows) >= 5:
        k = np.array([1, 2, 3, 2, 1], float) / 9
        deltas = np.stack([np.convolve(np.pad(deltas[:, c], 2, mode="edge"), k, mode="valid") for c in range(3)], 1)
    Y = np.arange(orig.shape[0], dtype=float)
    D = np.stack([np.interp(Y, rows, deltas[:, c]) for c in range(3)], 1)[:, None, :]
    wL = np.clip((Ln[..., 0] - (medf - 35)) / 15, 0, 1)
    Wt = np.clip(cv2.GaussianBlur(sn.astype(np.float32), (0, 0), max(3, .025 * h)) * 1.8, 0, 1) * cv2.GaussianBlur(wL, (0, 0), 3)
    Wt *= cv2.GaussianBlur(R.astype(np.float32), (0, 0), 5)
    out = cv2.cvtColor(np.clip(Ln + D * Wt[..., None], 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
    return np.where((Wt > 0.003)[..., None], out, final)


QA_LIMITS = {"face_tone": 6.0, "misjoin": 10.0}


def _bg_match(gen, orig, lo, lg, mask):
    """Where the new head is smaller than the old one, the model re-draws the background; match that re-drawn background
    to the real background right next to it (local brightness + colour), so no lighter/darker halo shows around the head."""
    import cv2, numpy as np
    res = gen.shape[0]
    ref = (lo == 0) & (mask < 0.1)                 # real background just outside the pasted area
    src = (lg == 0) & (mask > 0.3)                 # re-drawn background inside the pasted area
    if ref.sum() < 300 or src.sum() < 100:
        return gen, 0
    sig = 0.018 * res                               # very local: never mixes two different background areas
    O = cv2.cvtColor(orig, cv2.COLOR_BGR2LAB).astype(np.float32)
    G = cv2.cvtColor(gen, cv2.COLOR_BGR2LAB).astype(np.float32)

    def nc(img, m, s):
        mf = m.astype(np.float32)
        den = cv2.GaussianBlur(mf, (0, 0), s) + 1e-4
        return np.stack([cv2.GaussianBlur(img[..., c] * mf, (0, 0), s) / den for c in range(3)], -1), den

    fo, do_ = nc(O, ref, sig)
    fg, dg_ = nc(G, src, sig)
    ok = (do_ > 0.02) & (dg_ > 0.02)
    if ok.sum() < 100:
        return gen, 0
    d = np.where(ok[..., None], fo - fg, 0.0)
    corr, _ = nc(d, ok, sig)
    w = cv2.GaussianBlur(src.astype(np.float32), (0, 0), 0.006 * res)
    corr[..., 0] = np.clip(corr[..., 0], -12, 12)
    corr[..., 1:] = np.clip(corr[..., 1:], -5, 5)
    G += corr * w[..., None]
    return cv2.cvtColor(np.clip(G, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR), int(src.sum())


def _jaw_blend(gen, orig, lo, lg, mask, fy, face_w, max_shift=25.0):
    """Make the new face meet the ORIGINAL neck without a colour edge: measure the colour on both sides of the lower
    (jaw/chin) border of the pasted area, and correct the new skin near the jaw by the difference, fading out upwards."""
    import cv2, numpy as np
    res = gen.shape[0]
    inside = (mask > 0.5).astype(np.uint8)
    k = int(0.03 * res) | 1
    edge = cv2.morphologyEx(inside, cv2.MORPH_GRADIENT, np.ones((k, k), np.uint8)) > 0
    edge[: int(fy)] = False                                            # only the lower border (jaw / chin)
    s_o = edge & (inside == 0) & np.isin(lo, [1, 17])                  # original neck/skin just outside
    s_g = edge & (inside == 1) & np.isin(lg, [1, 2, 8, 9, 17])         # new skin just inside
    if s_o.sum() < 150 or s_g.sum() < 150:
        return gen, 0
    sig = max(3.0, 0.10 * face_w)
    O = cv2.cvtColor(orig, cv2.COLOR_BGR2LAB).astype(np.float32)
    G = cv2.cvtColor(gen, cv2.COLOR_BGR2LAB).astype(np.float32)

    def field(img, m):
        mf = m.astype(np.float32)
        den = cv2.GaussianBlur(mf, (0, 0), sig) + 1e-4
        return np.stack([cv2.GaussianBlur(img[..., c] * mf, (0, 0), sig) / den for c in range(3)], -1), den

    fo, do_ = field(O, s_o)
    fg, dg_ = field(G, s_g)
    ok = (do_ > 0.02) & (dg_ > 0.02) & edge
    if ok.sum() < 100:
        return gen, 0
    diff = fo - fg
    diff[..., 0] = np.clip(diff[..., 0], -max_shift, max_shift)          # brightness
    diff[..., 1:] = np.clip(diff[..., 1:] * 0.7, -8.0, 8.0)              # colour tint: gentler, no orange chin
    okf = ok.astype(np.float32)
    big = max(4.0, 0.22 * face_w)
    den = cv2.GaussianBlur(okf, (0, 0), big) + 1e-4
    corr = np.stack([cv2.GaussianBlur(diff[..., c] * okf, (0, 0), big) / den for c in range(3)], -1)
    dist = cv2.distanceTransform((~ok).astype(np.uint8), cv2.DIST_L2, 5)
    w = np.exp(-dist / max(4.0, 0.20 * face_w))                           # fades out before the mouth
    w *= cv2.GaussianBlur(np.isin(lg, [1, 2, 8, 9, 17]).astype(np.float32), (0, 0), 0.006 * res + 1)
    w *= _not_beard(G[..., 0], lg)         # not on beard / stubble
    G += corr * w[..., None]
    return cv2.cvtColor(np.clip(G, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR), int(ok.sum())


def _download(url, path):
    import requests
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    open(path, "wb").write(r.content)


def _iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def _target_embedding(fa, img):
    import cv2, numpy as np
    for pad in (0.35, 0.6, 0.0):
        h, w = img.shape[:2]
        s = max(1, int(640 / max(h, w)))
        big = cv2.resize(img, (w * s, h * s), interpolation=cv2.INTER_CUBIC)
        p = int(pad * max(big.shape[:2]))
        big = cv2.copyMakeBorder(big, p, p, p, p, cv2.BORDER_REPLICATE) if p else big
        fs = fa.get(big)
        if fs:
            f = max(fs, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
            return f.normed_embedding
    return None


@app.function(image=cpu_image, volumes={JOBS: jobs_vol}, cpu=4, memory=8192, timeout=1800)
def preprocess(template_id: str, video_url: str, target_face_url: str):
    import cv2, numpy as np
    t0 = time.time()
    d = f"{JOBS}/{template_id}"
    os.makedirs(d, exist_ok=True)
    _download(video_url, f"{d}/video.mp4")
    _download(target_face_url, f"{d}/target.png")
    fa = _fa()
    temb = _target_embedding(fa, cv2.imread(f"{d}/target.png"))
    cap = cv2.VideoCapture(f"{d}/video.mp4")
    fps = cap.get(cv2.CAP_PROP_FPS)
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    track, prev, i = [], None, 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        faces = fa.get(fr)
        best, bs = None, -9
        for f in faces:
            sim = float(np.dot(f.normed_embedding, temb)) if temb is not None else 0.0
            box = [float(x) for x in f.bbox]
            cont = _iou(prev, box) if prev else 0.0
            area = (box[2] - box[0]) * (box[3] - box[1]) / (W * H)
            score = sim + 0.6 * cont + (0.5 * area if temb is None else 0.0)
            if score > bs:
                best, bs = (box, sim, cont), score
        if best and (temb is None or best[1] >= 0.22 or best[2] >= 0.3):
            track.append({"i": i, "box": best[0], "sim": round(best[1], 3)})
            prev = best[0]
        else:
            track.append({"i": i, "box": None, "sim": round(best[1], 3) if best else None})
            prev = None
        i += 1
    cap.release()
    meta = {"template_id": template_id, "fps": fps, "w": W, "h": H, "frames": i, "target_emb_found": temb is not None,
            "frames_with_target": sum(1 for t in track if t["box"]), "track": track}
    json.dump(meta, open(f"{d}/track.json", "w"))
    jobs_vol.commit()
    return {k: v for k, v in meta.items() if k != "track"} | {"seconds": round(time.time() - t0, 1)}


def _smooth_track(track, n, max_gap=6, sigma=2.0):
    import numpy as np
    boxes = [t["box"] for t in track]
    # fill short gaps by linear interpolation
    idx = [i for i, b in enumerate(boxes) if b]
    out = list(boxes)
    for a, b in zip(idx, idx[1:]):
        if 1 < b - a <= max_gap + 1:
            for k in range(a + 1, b):
                w = (k - a) / (b - a)
                out[k] = [boxes[a][j] * (1 - w) + boxes[b][j] * w for j in range(4)]
    # gaussian smoothing within contiguous runs
    res, i = list(out), 0
    while i < n:
        if out[i] is None:
            i += 1; continue
        j = i
        while j < n and out[j] is not None:
            j += 1
        arr = np.array(out[i:j], dtype=float)
        rad = max(1, int(round(2.0 * sigma)))       # sigma 2 -> +-4 frames (old default); larger = steadier crop
        k = np.exp(-0.5 * (np.arange(-rad, rad + 1) / sigma) ** 2); k /= k.sum()
        pad = np.pad(arr, ((rad, rad), (0, 0)), mode="edge")
        sm = np.stack([np.convolve(pad[:, c], k, mode="valid") for c in range(4)], 1)
        for q in range(i, j):
            res[q] = sm[q - i].tolist()
        i = j
    return res


def _geom_base(box):
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    S = 2.6 * max(h, w * 1.15)          # crop side
    return cx, cy - 0.12 * h, S, w, h, cx, cy


_geom = _geom_base


def _crop(frame, cx, cy, S, out):
    import cv2, numpy as np
    sc = out / S
    M = np.array([[sc, 0, out / 2 - sc * cx], [0, sc, out / 2 - sc * cy]], dtype=np.float32)
    return cv2.warpAffine(frame, M, (out, out), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT_101), M


def _head_mask(out, S, w, h, cx, cy, ccx, ccy, ax=0.95, ay=1.15, dy=0.25, fe=0.14):
    """feathered ellipse in crop space"""
    import cv2, numpy as np
    sc = out / S
    ex, ey = out / 2 + sc * (cx - ccx), out / 2 + sc * ((cy - dy * h) - ccy)
    m = np.zeros((out, out), np.float32)
    cv2.ellipse(m, (int(ex), int(ey)), (int(ax * w * sc), int(ay * h * sc)), 0, 0, 360, 1.0, -1)
    k = int(fe * h * sc) * 2 + 1
    return cv2.GaussianBlur(m, (k, k), 0)


def _color_match(gen, orig, mask):
    import cv2, numpy as np
    ring = (mask < 0.02).astype(np.float32)
    if ring.sum() < 500:
        return gen
    g = cv2.cvtColor(gen, cv2.COLOR_BGR2LAB).astype(np.float32)
    o = cv2.cvtColor(orig, cv2.COLOR_BGR2LAB).astype(np.float32)
    r = ring.astype(bool)
    for c in range(3):
        gm, gs = g[..., c][r].mean(), g[..., c][r].std() + 1e-3
        om, os_ = o[..., c][r].mean(), o[..., c][r].std() + 1e-3
        g[..., c] = (g[..., c] - gm) * float(np.clip(os_ / gs, 0.85, 1.2)) + om
    return cv2.cvtColor(np.clip(g, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)


def _not_beard(L, lg):
    """1 on normal skin, 0 on beard / stubble / brows: anything clearly darker than the typical skin of this face
    (relative to the median skin brightness, so it works for light and dark skin and for mid-brown beards)"""
    import numpy as np
    sk = lg == 1
    med = float(np.median(L[sk])) if sk.sum() > 200 else 60.0
    import cv2
    w = np.clip((L - (med - 30.0)) / 14.0, 0.0, 1.0)          # only clearly dark hair (beard, brows), not skin shadows
    return cv2.GaussianBlur(w.astype(np.float32), (0, 0), max(2.0, 0.006 * L.shape[0]))   # smooth: no blotches


SKIN_CLASSES = [1, 2, 8, 9, 17]   # face skin, nose, ears, neck (face-parsing labels)


def _match_sharpness_grain(img, gw, m, seed=0):
    """Make the pasted head as sharp/soft and as grainy as the original photo, so it does not look pasted on.
    Compared at the final size inside the head area: detail = std of the Laplacian; grain = std of the finest residual."""
    import cv2, numpy as np
    ys, xs = np.nonzero(m > 0.05)
    if len(ys) < 500:
        return gw, {}
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    core = m[y0:y1, x0:x1] > 0.6
    if core.sum() < 300:
        return gw, {}
    go = cv2.cvtColor(img[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY).astype(np.float32)
    roi = gw[y0:y1, x0:x1].astype(np.float32)

    def detail(g):
        return float(cv2.Laplacian(g, cv2.CV_32F)[core].std())

    d_o = detail(go)
    d_g = detail(cv2.cvtColor(roi.clip(0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32))
    sig = 0.0
    if d_g > 1.15 * d_o:
        lo_, hi_ = 0.0, 6.0
        for _ in range(14):
            mid = (lo_ + hi_) / 2
            b = cv2.GaussianBlur(roi, (0, 0), mid) if mid > 0.05 else roi
            if detail(cv2.cvtColor(b.clip(0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)) > d_o:
                lo_ = mid
            else:
                hi_ = mid
        sig = hi_
        roi = cv2.GaussianBlur(roi, (0, 0), sig)
    # grain: add the missing fine noise (luma) so the head has the same texture as the photo
    def grain(g):
        return float((g - cv2.GaussianBlur(g, (0, 0), 1.0))[core].std())
    gr_o = grain(go)
    gr_g = grain(cv2.cvtColor(roi.clip(0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32))
    add = float(np.sqrt(max(0.0, gr_o ** 2 - gr_g ** 2)))
    if add > 0.3:
        rng = np.random.default_rng(seed)
        n = rng.normal(0, add, roi.shape[:2]).astype(np.float32)
        n = n - cv2.GaussianBlur(n, (0, 0), 1.0) * 0.5
        roi = roi + n[..., None]
    out = gw.copy()
    out[y0:y1, x0:x1] = roi.clip(0, 255).astype(np.uint8)
    return out, {"blur_sigma": round(sig, 2), "detail_orig": round(d_o, 1), "detail_new_before": round(d_g, 1), "grain_added": round(add, 2)}


def _skin_tone_match(gen, orig, sess, mask, strength=0.9):
    """Shift the generated head's skin (face, nose, ears, neck) to the skin tone of the original person, so the new face
    matches the neck, chest and hands that stay in the picture. Eyes, brows, lips, hair and jewellery are not changed.
    Compared like with like: skin + neck of the generated head vs skin + neck of the original (same light, same place)."""
    import cv2, numpy as np
    res = gen.shape[0]
    lo = cv2.resize(_parse_labels(sess, orig).astype(np.uint8), (res, res), interpolation=cv2.INTER_NEAREST)
    lg = cv2.resize(_parse_labels(sess, gen).astype(np.uint8), (res, res), interpolation=cv2.INTER_NEAREST)
    ref = np.isin(lo, [1, 17]) & (cv2.dilate((mask > 0.05).astype(np.uint8), np.ones((int(0.08 * res) | 1,) * 2, np.uint8)) > 0)
    src = np.isin(lg, [1, 17]) & (mask > 0.5)
    if ref.sum() < 400 or src.sum() < 400:
        return gen
    G = cv2.cvtColor(gen, cv2.COLOR_BGR2LAB).astype(np.float32)
    O = cv2.cvtColor(orig, cv2.COLOR_BGR2LAB).astype(np.float32)
    w = cv2.GaussianBlur(np.isin(lg, SKIN_CLASSES).astype(np.float32), (0, 0), 0.008 * res) * strength
    w = w * _not_beard(G[..., 0], lg)      # dark pixels (beard, stubble, brows) are not tinted
    for c in range(3):
        d = float(np.median(O[..., c][ref]) - np.median(G[..., c][src]))
        G[..., c] += d * w
    return cv2.cvtColor(np.clip(G, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)


def _cm_params(gen, orig, mask):
    """same transfer as _color_match, returned as per-channel LAB (gain, offset) so it can be smoothed over time"""
    import cv2, numpy as np
    r = mask < 0.02
    if r.sum() < 500:
        return None
    g = cv2.cvtColor(gen, cv2.COLOR_BGR2LAB).astype(np.float32)
    o = cv2.cvtColor(orig, cv2.COLOR_BGR2LAB).astype(np.float32)
    p = np.zeros((3, 2), np.float32)
    for c in range(3):
        gm, gs = g[..., c][r].mean(), g[..., c][r].std() + 1e-3
        om, os_ = o[..., c][r].mean(), o[..., c][r].std() + 1e-3
        a = float(np.clip(os_ / gs, 0.85, 1.2))
        p[c] = (a, om - gm * a)
    return p


def _cm_apply(gen, p):
    import cv2, numpy as np
    if p is None:
        return gen
    g = cv2.cvtColor(gen, cv2.COLOR_BGR2LAB).astype(np.float32)
    for c in range(3):
        g[..., c] = g[..., c] * p[c, 0] + p[c, 1]
    return cv2.cvtColor(np.clip(g, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)


@app.function(image=cpu_image, volumes={JOBS: jobs_vol}, cpu=4, memory=16384, timeout=3600)
def swap(template_id: str, source_face_url: str, start_frame: int = 0, max_frames: int = 0, max_cost_usd: float = 1.5,
         seed: int = 42, steps: int = 8, sampler: str = "euler", dry_run: bool = False):
    import cv2, numpy as np
    t0 = time.time()
    jobs_vol.reload()
    d = f"{JOBS}/{template_id}"
    meta = json.load(open(f"{d}/track.json"))
    n, fps, W, H = meta["frames"], meta["fps"], meta["w"], meta["h"]
    end = n if not max_frames else min(n, start_frame + max_frames)
    sm = _smooth_track(meta["track"], n)
    todo = [i for i in range(start_frame, end) if sm[i] is not None]
    est_gpu_s = len(todo) * 11.0 + 4 * 110
    est_cost = est_gpu_s * GPU_USD_PER_SEC
    info = {"frames_total": n, "frames_range": [start_frame, end], "frames_to_process": len(todo), "est_cost_usd": round(est_cost, 2)}
    if dry_run or est_cost > max_cost_usd:
        return info | {"aborted": not dry_run, "reason": None if dry_run else f"estimated cost ${est_cost:.2f} > cap ${max_cost_usd}"}

    # --- head reference crop from the source photo
    sp = f"{d}/source_{uuid.uuid4().hex[:6]}.jpg"
    _download(source_face_url, sp)
    src = cv2.imread(sp)
    fa = _fa()
    fs = fa.get(src)
    if not fs:
        raise RuntimeError("no face found in source photo")
    f = max(fs, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    x0, y0, x1, y1 = f.bbox
    cx, cy, S = (x0 + x1) / 2, (y0 + y1) / 2 - 0.2 * (y1 - y0), 2.3 * max(y1 - y0, (x1 - x0) * 1.15)
    head_crop, _ = _crop(src, cx, cy, S, HEAD_RES)
    ok, hb = cv2.imencode(".png", head_crop)
    head_png = hb.tobytes()

    # --- read frames, build crops
    cap = cv2.VideoCapture(f"{d}/video.mp4")
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frames, crops, geoms = {}, [], {}
    for i in range(start_frame, end):
        ok, fr = cap.read()
        if not ok:
            break
        frames[i] = fr
        if sm[i] is not None:
            cx_, cy_, S_, w_, h_, bx, by = _geom(sm[i])
            crop, M = _crop(fr, cx_, cy_, S_, BODY_RES)
            geoms[i] = (cx_, cy_, S_, w_, h_, bx, by, M, crop)
            crops.append((i, cv2.imencode(".png", crop)[1].tobytes()))
    cap.release()

    # --- GPU
    BATCH = 12
    batches = [crops[k:k + BATCH] for k in range(0, len(crops), BATCH)]
    outs = {}
    tg = time.time()
    for res in Worker().process.map(batches, kwargs={"head_png": head_png, "steps": steps, "sampler": sampler, "seed": seed}):
        for idx, png in res:
            outs[idx] = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)
    gpu_wall = time.time() - tg

    # --- composite + encode
    out_path = f"{d}/out_{uuid.uuid4().hex[:6]}.mp4"
    vstart = start_frame / fps
    dur = (end - start_frame) / fps
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(fps), "-i", "-",
           "-ss", f"{vstart:.3f}", "-t", f"{dur:.3f}", "-i", f"{d}/video.mp4", "-map", "0:v", "-map", "1:a?",
           "-c:v", "libx264", "-crf", "16", "-preset", "medium", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k", "-shortest", out_path]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    sample_dir = f"{d}/samples_{os.path.basename(out_path)[4:-4]}"
    os.makedirs(sample_dir, exist_ok=True)
    for i in range(start_frame, start_frame + len(frames)):
        fr = frames[i]
        if i in outs:
            cx_, cy_, S_, w_, h_, bx, by, M, crop = geoms[i]
            gen = outs[i]
            if gen.shape[:2] != crop.shape[:2]:
                gen = cv2.resize(gen, crop.shape[:2][::-1], interpolation=cv2.INTER_CUBIC)
            mask = _head_mask(BODY_RES, S_, w_, h_, bx, by, cx_, cy_)
            gen = _color_match(gen, crop, mask)
            inv = cv2.invertAffineTransform(M)
            gw = cv2.warpAffine(gen, inv, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            mw = cv2.warpAffine(mask, inv, (W, H), flags=cv2.INTER_LINEAR, borderValue=0)[..., None]
            fr = (fr.astype(np.float32) * (1 - mw) + gw.astype(np.float32) * mw).clip(0, 255).astype(np.uint8)
            if (i - start_frame) % 40 == 0:
                cv2.imwrite(f"{sample_dir}/f{i:04d}.jpg", fr, [cv2.IMWRITE_JPEG_QUALITY, 92])
        p.stdin.write(fr.tobytes())
    p.stdin.close(); p.wait()
    jobs_vol.commit()
    return info | {"output": out_path.replace(JOBS + "/", ""), "processed": len(outs), "gpu_wall_s": round(gpu_wall, 1),
                   "total_s": round(time.time() - t0, 1), "model": "Qwen-Image-2.1 + bfs_head_v1.1_qwen_2.1"}


def _res_for_size(S, cap):
    """Generation size for a head crop that covers S pixels of the source: ~1.25x its on-screen size, multiple of 64."""
    import math
    return int(min(cap, max(MIN_RES, math.ceil(1.25 * S / 64) * 64)))


def _adaptive_res(template_id, start_frame, max_frames, cap):
    """One size for the whole job (changing it per frame would add flicker), from the median head-crop size."""
    import numpy as np
    jobs_vol.reload()
    meta = json.load(open(f"{JOBS}/{template_id}/track.json"))
    n = meta["frames"]
    end = n if not max_frames else min(n, start_frame + max_frames)
    sm = _smooth_track(meta["track"], n)
    sizes = [_geom_base(sm[i])[2] for i in range(start_frame, end) if sm[i] is not None]
    return _res_for_size(float(np.median(sizes)), cap) if sizes else cap


def _cleanup(keep_days=KEEP_DAYS):
    """Delete old outputs / uploads / temp folders so storage stays inside the free allowance.
    Templates (video.mp4 + track.json) are kept so they can be re-used."""
    import shutil
    cut = time.time() - keep_days * 86400
    removed = 0
    for root, dirs, files in os.walk(JOBS):
        for nm in list(dirs):
            p = os.path.join(root, nm)
            if (nm.startswith("raw_") or nm.startswith("parts_")) and os.path.getmtime(p) < time.time() - 86400:
                shutil.rmtree(p, ignore_errors=True)
                dirs.remove(nm)
                removed += 1
        for nm in files:
            if nm in ("video.mp4", "track.json"):
                continue
            p = os.path.join(root, nm)
            if os.path.getmtime(p) < cut:
                os.remove(p)
                removed += 1
    return removed


@app.function(image=cpu_image, volumes={JOBS: jobs_vol}, cpu=4, memory=4096, timeout=5400)
def swap_job(template_id: str, source_face_url: str, quality: str = "best", start_frame: int = 0, max_frames: int = 0,
             max_cost_usd: float = 5.0, seed: int = 42, steps: int = 8, sampler: str = "euler", output_put_url: str = "",
             dry_run: bool = False, keep_raw: bool = False):
    """Video head swap: GPU stage once, then blending split over many small CPU containers, then join + audio."""
    import shutil
    t0 = time.time()
    preset = dict(PRESETS.get(quality, PRESETS["best"]))
    if preset.pop("adaptive", False):
        preset["res"] = _adaptive_res(template_id, start_frame, max_frames, preset["res"])
    denoise = preset.pop("denoise", False)
    common = dict(seed=seed, steps=steps, sampler=sampler, seg_mask=True, **preset)
    g = swap2.remote(template_id, source_face_url, start_frame=start_frame, max_frames=max_frames, max_cost_usd=max_cost_usd,
                     dry_run=dry_run, gpu_only=True, **common)
    if dry_run or g.get("aborted"):
        return g | {"quality": quality, "res": preset["res"]}
    raw, fps, end = g["raw_dir"], g["fps"], g["end"]
    tag = raw.replace("raw_", "")
    calls = []
    for a in range(start_frame, end, BLEND_CHUNK):
        b = min(end, a + BLEND_CHUNK)
        s0 = max(start_frame, a - BLEND_PAD)
        e0 = min(end, b + BLEND_PAD)
        calls.append(blend_part.spawn(template_id, source_face_url, start_frame=s0, max_frames=e0 - s0, reuse_raw=raw,
                                      core=[a, b], out_name=f"parts_{tag}/{a:06d}.mp4", max_cost_usd=99, **common))
    parts = [c.get()["part"] for c in calls]
    jobs_vol.reload()
    d = f"{JOBS}/{template_id}"
    lst = f"{d}/parts_{tag}/list.txt"
    open(lst, "w").write("".join(f"file '{d}/{p}'\n" for p in parts))
    out_path = f"{d}/out2_{tag}.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", lst,
                    "-ss", f"{start_frame / fps:.3f}", "-t", f"{(end - start_frame) / fps:.3f}", "-i", f"{d}/video.mp4",
                    "-map", "0:v", "-map", "1:a?",
                    *(["-vf", "hqdn3d=0:0:10:12", "-c:v", "libx264", "-crf", "16", "-preset", "medium", "-pix_fmt", "yuv420p"]
                      if denoise else ["-c:v", "copy"]),
                    "-c:a", "aac", "-b:a", "160k", "-shortest", out_path], check=True)
    shutil.rmtree(f"{d}/parts_{tag}", ignore_errors=True)
    if not keep_raw:
        shutil.rmtree(f"{d}/{raw}", ignore_errors=True)
    try:
        cleaned = _cleanup()
    except Exception:
        cleaned = -1
    jobs_vol.commit()
    uploaded = _put_file(out_path, output_put_url, "video/mp4") if output_put_url else None
    return {k: g[k] for k in ("frames_range", "head_frames", "key_frames", "est_cost_usd", "gpu_wall_s") if k in g} | {
        "output": out_path.replace(JOBS + "/", ""), "quality": quality, "res": preset["res"], "blend_parts": len(parts),
        "total_s": round(time.time() - t0, 1), "uploaded": uploaded, "old_files_removed": cleaned}


# ----------------------------------------------------------------------------- HTTP API
@app.function(image=cpu_image, volumes={JOBS: jobs_vol}, timeout=600)
@modal.asgi_app(label="headswap-studio-basic")
def web():
    from fastapi import FastAPI, Header, HTTPException, Request
    from fastapi.responses import FileResponse
    api = FastAPI(title="BFS head-swap")

    REQUIRE_AUTH = False   # set True to require the Bearer token again

    def auth(a):
        if REQUIRE_AUTH and a != f"Bearer {os.environ['API_TOKEN']}":
            raise HTTPException(401, "bad token")

    @api.post("/preprocess")
    def pre(body: dict, authorization: str = Header(None)):
        auth(authorization)
        fc = preprocess.spawn(body["template_id"], body["video_url"], body["target_face_url"])
        return {"job_id": fc.object_id, "status": "running"}

    @api.post("/swap")
    def sw(body: dict, authorization: str = Header(None)):
        auth(authorization)
        jobs_vol.reload()
        if not os.path.exists(f"{JOBS}/{body['template_id']}/track.json"):
            raise HTTPException(404, "template not preprocessed: call /preprocess first")
        kw = {k: body[k] for k in ("quality", "start_frame", "max_frames", "max_cost_usd", "seed", "steps", "sampler", "dry_run", "output_put_url") if k in body}
        fc = swap_job.spawn(body["template_id"], body["source_face_url"], **kw)
        return {"job_id": fc.object_id, "status": "running"}

    @api.post("/swap-video")
    def swap_video_ep(body: dict, authorization: str = Header(None)):
        auth(authorization)
        kw = {k: body[k] for k in ("template_id", "video_url", "target_face_url", "quality", "max_cost_usd", "output_put_url") if k in body}
        fc = swap_video.spawn(body["source_face_url"], **kw)
        return {"job_id": fc.object_id, "status": "running"}

    @api.post("/swap-image")
    def swap_image_ep(body: dict, authorization: str = Header(None)):
        auth(authorization)
        kw = {k: body[k] for k in ("target_face_url", "face_index", "quality", "seed", "output_put_url", "max_cost_usd", "skin_match", "keep_neck", "protect_cloth", "match_sharpness", "best_of", "relight", "jaw_blend", "harmonize", "steps", "cfg", "distill", "neck_fade", "bg_match", "neck_depth", "engine", "prompt", "lora") if k in body}
        fc = swap_image.spawn(body["image_url"], body["source_face_url"], **kw)
        return {"job_id": fc.object_id, "status": "running"}

    @api.get("/jobs/{job_id}")
    def job(job_id: str, request: Request, authorization: str = Header(None)):
        auth(authorization)
        try:
            res = modal.FunctionCall.from_id(job_id).get(timeout=0)
            if isinstance(res, dict) and res.get("output"):
                base = str(request.base_url).rstrip("/").replace("http://", "https://")
                res["download_url"] = f"{base}/download/{res['output']}"
            return {"status": "done", "result": res}
        except TimeoutError:
            return {"status": "running"}
        except Exception as e:
            return {"status": "error", "error": str(e)[:800]}

    @api.get("/download/{path:path}")
    def dl(path: str, authorization: str = Header(None)):
        auth(authorization)
        jobs_vol.reload()
        full = os.path.normpath(os.path.join(JOBS, path))
        if not full.startswith(JOBS + os.sep) or not os.path.isfile(full):
            raise HTTPException(404)
        return FileResponse(full)

    return api


# ----------------------------------------------------------------------------- PRO head-swap API (best quality, one call)
# engine 'bfs' (Qwen Image 2.1 + BFS head v1.1) tested best on the Blenders Pride ads (identity 0.80-0.89); 'edit'
# (Qwen-Image-Edit-2511 + BFS V5) is available as an option - it is prompt-controllable but gave harsher/plastic skin there
@app.function(image=cpu_image, volumes={JOBS: jobs_vol}, cpu=4, memory=16384, timeout=3600)
def studio_job(image_url: str, face_url: str, kw: dict, max_rounds: int = 2):
    """Quality-gated head swap: generate (best-of-N), measure (clothes untouched, face tone vs original model, jaw->neck
    join vs original), try the neck-profile repair, and if the result still fails the limits, generate again with new
    seeds. Returns the best result plus its quality numbers and a pass flag."""
    import cv2, numpy as np
    jobs_vol.reload()
    jid = uuid.uuid4().hex[:10]
    os.makedirs(f"{JOBS}/images", exist_ok=True)
    _download(image_url, f"{JOBS}/images/{jid}_orig")
    orig = cv2.imread(f"{JOBS}/images/{jid}_orig")
    tried, best = [], None
    for rnd in range(max_rounds):
        k = dict(kw)
        k["seed"] = int(kw.get("seed", 21)) + 50000 * rnd
        r = swap_image.remote(image_url, face_url, **k)
        jobs_vol.reload()
        fin = cv2.imread(f"{JOBS}/{r['output']}")
        for name, img in (("base", fin), ("neck_profile_fix", _profile_fix(orig, fin))):
            q = _qa(orig, img)
            q["passed"] = (q["clothes_px"] == 0 and abs(q["face_tone"]) <= QA_LIMITS["face_tone"] and q["misjoin"] <= QA_LIMITS["misjoin"])
            tried.append({"round": rnd + 1, "variant": name, "seed": r.get("seed_used")} | q)
            if best is None or (q["passed"], -q["score"]) > (best[1]["passed"], -best[1]["score"]):
                best = (img, q, rnd + 1, name, r)
        if best[1]["passed"]:
            break
    img, q, rnd, name, r = best
    outp = f"{JOBS}/images/{jid}_studio.png"
    cv2.imwrite(outp, img)
    jobs_vol.commit()
    return {"output": f"images/{jid}_studio.png", "passed_quality_gate": q["passed"], "quality": q, "chosen": {"round": rnd, "variant": name},
            "all_tries": tried, "limits": QA_LIMITS, "swap": {k2: r.get(k2) for k2 in ("code", "engine", "seed_used", "tone", "candidates")}}


PRO_DEFAULTS = dict(quality="best", engine="klein", best_of=4, steps=4, lora=1.0, skin_match=1.0, keep_neck=False, jaw_blend=False,
                    neck_depth=1.0, neck_fade=0.22, protect_cloth=True, bg_match=True, relight=0.45, match_sharpness=False,
                    harmonize=0.0, seed=21, max_cost_usd=1.0, tone_finish=True)


@app.function(image=cpu_image, volumes={JOBS: jobs_vol}, timeout=600)
@modal.asgi_app(label="headswap-studio-api")
def pro_api():
    """Best-quality image head swap. Engine: open-source Qwen-Image-Edit-2511 + BFS head V5 (instruction editing), followed by
    the finishing pipeline: best-of-N selection (identity, expression, head pose, mouth, extra hair), head+neck paste onto the
    ORIGINAL image, clothes / jewellery protection, old-hair removal, skin-tone + light match, neck transition, background match."""
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import FileResponse
    api = FastAPI(title="Head swap PRO (Qwen-Image-Edit-2511 + BFS V5)")

    @api.post("/headswap")
    def headswap(body: dict):
        """body: image_url (poster), face_url (new face). Optional: target_face_url / face_index (which person),
        prompt_extra (extra instructions, e.g. 'nude lips, closed mouth'), best_of (1-6), neck_depth (0.76 if a necklace sits
        on the neck, 1.0-1.2 otherwise), engine ('edit' or 'bfs'), seed."""
        if not body.get("image_url") or not body.get("face_url"):
            raise HTTPException(400, "image_url and face_url are required")
        kw = dict(PRO_DEFAULTS)
        for k in ("target_face_url", "face_index", "best_of", "neck_depth", "engine", "seed", "lora", "relight", "skin_match", "steps",
                  "distill", "cfg"):
            if k in body:
                kw[k] = body[k]
        kw["best_of"] = max(1, min(6, int(kw["best_of"])))
        if body.get("prompt_extra") or body.get("prompt"):
            kw["prompt"] = body.get("prompt") or (EDIT_PROMPT + EDIT_PROMPT_EXTRA + " " + body["prompt_extra"])
        rounds = max(1, min(3, int(body.get("max_rounds", 2))))
        fc = studio_job.spawn(body["image_url"], body["face_url"], kw, rounds)
        return {"job_id": fc.object_id, "status": "running", "settings": {k: v for k, v in kw.items() if k != "prompt"} | {"max_rounds": rounds}}

    @api.get("/jobs/{job_id}")
    def job(job_id: str, request: Request):
        try:
            res = modal.FunctionCall.from_id(job_id).get(timeout=0)
            if isinstance(res, dict) and res.get("output"):
                base = str(request.base_url).rstrip("/").replace("http://", "https://")
                res["download_url"] = f"{base}/download/{res['output']}"
            return {"status": "done", "result": res}
        except TimeoutError:
            return {"status": "running"}
        except Exception as e:
            return {"status": "error", "error": str(e)[:800]}

    @api.get("/download/{path:path}")
    def dl(path: str):
        jobs_vol.reload()
        full = os.path.normpath(os.path.join(JOBS, path))
        if not full.startswith(JOBS + os.sep) or not os.path.isfile(full):
            raise HTTPException(404)
        return FileResponse(full)

    return api


# ----------------------------------------------------------------------------- temporal stabilisation (CPU only)
@app.function(image=cpu_image, volumes={JOBS: jobs_vol}, cpu=4, memory=16384, timeout=3000)
def stabilize(template_id: str, out_name: str, strength: float = 0.5):
    """Blend each swapped head with motion-aligned neighbours to cut frame-to-frame shimmer. No GPU."""
    import cv2, numpy as np
    from collections import deque
    jobs_vol.reload()
    d = f"{JOBS}/{template_id}"
    meta = json.load(open(f"{d}/track.json"))
    n, fps, W, H = meta["frames"], meta["fps"], meta["w"], meta["h"]
    sm = _smooth_track(meta["track"], n)
    co, cs = cv2.VideoCapture(f"{d}/video.mp4"), cv2.VideoCapture(f"{d}/{out_name}")
    outp = f"{d}/stab_{uuid.uuid4().hex[:6]}.mp4"
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(fps), "-i", "-",
           "-i", f"{d}/video.mp4", "-map", "0:v", "-map", "1:a?", "-c:v", "libx264", "-crf", "16", "-preset", "medium",
           "-pix_fmt", "yuv420p", "-c:a", "copy", "-shortest", outp]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    def read2():
        a, b = co.read(), cs.read()
        return (a[1], b[1]) if a[0] and b[0] else None

    win = deque()  # entries: (orig, out)
    for _ in range(2):
        r = read2()
        if r: win.append(r)
    t = 0
    prev_pair = None
    stats = {"blended": 0}
    while win:
        cur = win[0] if prev_pair is None else win[0]
        # window indexes: prev_pair (t-1), win[0] (t), win[1] (t+1)
        o_t, s_t = win[0]
        res = s_t
        if sm[t] is not None:
            cx_, cy_, S_, w_, h_, bx, by = _geom(sm[t])
            x0, y0 = int(max(0, cx_ - S_ / 2)), int(max(0, cy_ - S_ / 2))
            x1, y1 = int(min(W, cx_ + S_ / 2)), int(min(H, cy_ + S_ / 2))
            sc = 0.5
            def roi(a): return a[y0:y1, x0:x1]
            g_t = cv2.cvtColor(cv2.resize(roi(o_t), None, fx=sc, fy=sc), cv2.COLOR_BGR2GRAY)
            acc, wsum = roi(s_t).astype(np.float32) * 1.0, 1.0
            ref = cv2.GaussianBlur(roi(s_t), (0, 0), 3).astype(np.float32)
            for nb in (prev_pair, win[1] if len(win) > 1 else None):
                if nb is None: continue
                o_n, s_n = nb
                g_n = cv2.cvtColor(cv2.resize(roi(o_n), None, fx=sc, fy=sc), cv2.COLOR_BGR2GRAY)
                fl = cv2.calcOpticalFlowFarneback(g_t, g_n, None, 0.5, 3, 21, 3, 5, 1.1, 0)
                fl = cv2.resize(fl, (x1 - x0, y1 - y0)) / sc
                gx, gy = np.meshgrid(np.arange(x1 - x0, dtype=np.float32), np.arange(y1 - y0, dtype=np.float32))
                warped = cv2.remap(roi(s_n), gx + fl[..., 0], gy + fl[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                # validity: ignore pixels where the aligned neighbour disagrees with the current frame (occlusion / bad flow)
                diff = np.abs(cv2.GaussianBlur(warped, (0, 0), 3).astype(np.float32) - ref).mean(2)
                wv = (diff < 18).astype(np.float32)[..., None] * 0.5
                acc += warped.astype(np.float32) * wv
                wsum = wsum + wv
            blended = acc / wsum
            # feather by the head mask (frame space ROI)
            mask = _head_mask(BODY_RES, S_, w_, h_, bx, by, cx_, cy_)
            _, M = _crop(o_t, cx_, cy_, S_, BODY_RES)
            mw = cv2.warpAffine(mask, cv2.invertAffineTransform(M), (W, H), flags=cv2.INTER_LINEAR, borderValue=0)[y0:y1, x0:x1, None]
            mw = mw * strength * 2 if strength <= 0.5 else mw
            mw = np.clip(mw, 0, 1)
            new = roi(s_t).astype(np.float32) * (1 - mw) + blended * mw
            res = s_t.copy()
            res[y0:y1, x0:x1] = new.clip(0, 255).astype(np.uint8)
            stats["blended"] += 1
        p.stdin.write(res.tobytes())
        prev_pair = win.popleft()
        r = read2()
        if r: win.append(r)
        t += 1
    p.stdin.close(); p.wait()
    jobs_vol.commit()
    return {"output": outp.replace(JOBS + "/", ""), "frames": t, **stats}


# ----------------------------------------------------------------------------- v2: key-frame generation + motion-aligned interpolation
def _frame_mask(o_t, geom, W, H, **kw):
    import cv2
    cx_, cy_, S_, w_, h_, bx, by = geom
    mask = _head_mask(BODY_RES, S_, w_, h_, bx, by, cx_, cy_, **kw)
    _, M = _crop(o_t, cx_, cy_, S_, BODY_RES)
    return cv2.warpAffine(mask, cv2.invertAffineTransform(M), (W, H), flags=cv2.INTER_LINEAR, borderValue=0)[..., None]


def _aligned(o_t, o_n, s_n, box, sc=0.5):
    import cv2, numpy as np
    x0, y0, x1, y1 = box
    g_t = cv2.cvtColor(cv2.resize(o_t[y0:y1, x0:x1], None, fx=sc, fy=sc), cv2.COLOR_BGR2GRAY)
    g_n = cv2.cvtColor(cv2.resize(o_n[y0:y1, x0:x1], None, fx=sc, fy=sc), cv2.COLOR_BGR2GRAY)
    fl = cv2.calcOpticalFlowFarneback(g_t, g_n, None, 0.5, 3, 21, 3, 5, 1.1, 0)
    fl = cv2.resize(fl, (x1 - x0, y1 - y0)) / sc
    gx, gy = np.meshgrid(np.arange(x1 - x0, dtype=np.float32), np.arange(y1 - y0, dtype=np.float32))
    return cv2.remap(s_n[y0:y1, x0:x1], gx + fl[..., 0], gy + fl[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


_RAFT = {}


def _raft_flow(a_bgr, b_bgr):
    """dense flow a -> b (a(p) ~ b(p + f)) with RAFT-large (torchvision, BSD) on CPU; much more accurate than
    Farneback for large motion / long frame gaps"""
    import numpy as np, torch
    import torch.nn.functional as F
    from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
    if "m" not in _RAFT:
        torch.set_num_threads(max(1, os.cpu_count() or 4))
        _RAFT["m"] = raft_large(weights=Raft_Large_Weights.DEFAULT).eval()
    h, w = a_bgr.shape[:2]
    H8, W8 = (h + 7) // 8 * 8, (w + 7) // 8 * 8

    def prep(x):
        t = torch.from_numpy(np.ascontiguousarray(x[..., ::-1])).permute(2, 0, 1).float()[None] / 127.5 - 1.0
        return F.pad(t, (0, W8 - w, 0, H8 - h), mode="replicate")
    with torch.no_grad():
        fl = _RAFT["m"](prep(a_bgr), prep(b_bgr), num_flow_updates=12)[-1][0]
    return fl[:, :h, :w].permute(1, 2, 0).numpy()


def _flow_maps(o_t, o_n, box, sc=0.5, method="farneback"):
    """remap grids that pull pixels of frame n onto frame t inside box (ROI coordinates)"""
    import cv2, numpy as np
    x0, y0, x1, y1 = box
    a = cv2.resize(o_t[y0:y1, x0:x1], None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
    b = cv2.resize(o_n[y0:y1, x0:x1], None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
    if method == "raft":
        fl = _raft_flow(a, b)
    else:
        fl = cv2.calcOpticalFlowFarneback(cv2.cvtColor(a, cv2.COLOR_BGR2GRAY), cv2.cvtColor(b, cv2.COLOR_BGR2GRAY),
                                          None, 0.5, 3, 21, 3, 5, 1.1, 0)
    fl = cv2.resize(fl, (x1 - x0, y1 - y0)) / sc
    gx, gy = np.meshgrid(np.arange(x1 - x0, dtype=np.float32), np.arange(y1 - y0, dtype=np.float32))
    return (gx + fl[..., 0]).astype(np.float32), (gy + fl[..., 1]).astype(np.float32)



HEAD_CLASSES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17]   # skin..lips, hair, hat, earring, neck (not cloth)
_SEG = {}


def _seg_session(full=False):
    import onnxruntime as ort
    key = "full" if full else "q"
    if key not in _SEG:
        so = ort.SessionOptions()
        so.intra_op_num_threads = 4
        _SEG[key] = ort.InferenceSession("/seg/model.onnx" if full else "/seg/model_quantized.onnx", so, providers=["CPUExecutionProvider"])
    return _SEG[key]


def _parse_labels(sess, bgr):
    import cv2, numpy as np
    x = cv2.resize(bgr, (512, 512), interpolation=cv2.INTER_LINEAR)[..., ::-1].astype(np.float32) / 255.0
    x = ((x - np.array([0.485, 0.456, 0.406], np.float32)) / np.array([0.229, 0.224, 0.225], np.float32)).transpose(2, 0, 1)[None].astype(np.float32)
    lg = sess.run(None, {sess.get_inputs()[0].name: x})[0][0]
    lg = np.stack([cv2.resize(c, (512, 512), interpolation=cv2.INTER_LINEAR) for c in lg])
    return lg.argmax(0)


def _head_region(sess, bgr, fx, fy, fw, fh, cap, classes=None):
    """soft mask (0..1, same size as bgr) of the head+hair that contains the target face centre"""
    import cv2, numpy as np
    res = bgr.shape[0]
    lab = _parse_labels(sess, bgr)
    head = np.isin(lab, HEAD_CLASSES if classes is None else classes).astype(np.uint8)
    head = head * cv2.resize(cap.astype(np.uint8), (512, 512), interpolation=cv2.INTER_NEAREST)
    n, cc = cv2.connectedComponents(head, connectivity=8)
    sc = 512.0 / res
    core = np.zeros((512, 512), np.uint8)
    cv2.ellipse(core, (int(fx * sc), int(fy * sc)), (max(2, int(0.30 * fw * sc)), max(2, int(0.30 * fh * sc))), 0, 0, 360, 1, -1)
    keep = np.unique(cc[(core > 0) & (cc > 0)])
    m = np.isin(cc, keep).astype(np.float32) if len(keep) else np.zeros((512, 512), np.float32)
    m = cv2.GaussianBlur(m, (0, 0), 1.5)
    return cv2.resize(m, (res, res), interpolation=cv2.INTER_LINEAR)


TIGHT = dict(ax=0.72, ay=0.90, dy=0.24, fe=0.08)


@app.function(image=cpu_image, volumes={JOBS: jobs_vol}, cpu=4, memory=24576, timeout=3600)
def swap2(*a, **kw):
    return _swap2_impl(*a, **kw)


@app.function(image=cpu_image, volumes={JOBS: jobs_vol}, cpu=4, memory=8192, timeout=1800, max_containers=20)
def blend_part(*a, **kw):
    """Blends one slice of the video from already generated heads (reuse_raw) - many run in parallel."""
    return _swap2_impl(*a, **kw)


def _swap2_impl(template_id: str, source_face_url: str, start_frame: int = 0, max_frames: int = 0, res: int = 896, href: int = 704,
          max_cost_usd: float = 1.2, seed: int = 42, steps: int = 8, sampler: str = "euler", dry_run: bool = False,
          reuse_raw: str = "", promote_thr: float = 6.0, key_step: int = 2, grain: bool = False, grain_max: float = 2.5, fix_top: bool = True, align_fix: bool = False, align_sigma: float = 4.0, taps: int = 1, seg_mask: bool = True, seg_full: bool = False, output_put_url: str = "", flow_comp: bool = False, soft_match: bool = False, seq_ref: int = 0,
                gpu_only: bool = False, core: list = None, out_name: str = "",
                post_taps: int = 0, post_thr: float = 14.0, post_w: float = 0.6,
                stab_iters: int = 0, stab_taps: int = 2, stab_w: float = 0.8, cm_smooth: float = 0.0,
                prop: bool = False, flow: str = "farneback", prop_thr: float = 7.0, prop_sc: float = 0.5,
                prop_gate: float = 25.0, track_sigma: float = 2.0, vae_up: bool = False, gen_res: int = 0):
    import cv2, numpy as np
    t0 = time.time()
    jobs_vol.reload()
    d = f"{JOBS}/{template_id}"
    meta = json.load(open(f"{d}/track.json"))
    n, fps, W, H = meta["frames"], meta["fps"], meta["w"], meta["h"]
    end = n if not max_frames else min(n, start_frame + max_frames)
    sm = _smooth_track(meta["track"], n, sigma=track_sigma)
    todo = [i for i in range(start_frame, end) if sm[i] is not None]
    runs, cur = [], []
    for i in todo:
        if cur and i != cur[-1] + 1:
            runs.append(cur)
            cur = []
        cur.append(i)
    if cur:
        runs.append(cur)

    # read all frames of the range
    cap = cv2.VideoCapture(f"{d}/video.mp4")
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frames = {}
    for i in range(start_frame, end):
        ok, fr = cap.read()
        if not ok:
            break
        frames[i] = fr
    cap.release()

    def _geom(box):
        g = _geom_base(box)
        cx_, cy_, S_ = g[0], g[1], g[2]
        if S_ <= W:
            cx_ = min(max(cx_, S_ / 2), W - S_ / 2)
        if S_ <= H:
            cy_ = min(max(cy_, S_ / 2), H - S_ / 2)
        return (cx_, cy_, S_) + tuple(g[3:])

    def roi_box(geom):
        cx_, cy_, S_ = geom[0], geom[1], geom[2]
        return (int(max(0, cx_ - S_ / 2)), int(max(0, cy_ - S_ / 2)), int(min(W, cx_ + S_ / 2)), int(min(H, cy_ + S_ / 2)))

    # key frames: every key_step-th frame, plus any frame where motion-aligned interpolation would NOT reproduce the real footage
    keys, errs = [], {}
    FM = {}

    def fmaps(t_, k_, box_):
        if (t_, k_) not in FM:
            FM[(t_, k_)] = _flow_maps(frames[t_], frames[k_], box_, prop_sc, flow)
        return FM[(t_, k_)]

    if prop:
        # keyframe propagation: BFS heads only on a fixed absolute grid (identical in every parallel slice) + the
        # real start/end of each shot; frames in between are carried from the nearest keys by the original footage's
        # motion. A frame becomes a key itself when warping the ORIGINAL key frame onto it does not reproduce it
        # (mouth opening, eyes closing, occlusion, fast turns) - checked sequentially, so later frames use it.
        for r in runs:
            ks = {i for i in r if i % key_step == 0}
            if r[0] == 0 or sm[r[0] - 1] is None:
                ks.add(r[0])
            if r[-1] == n - 1 or sm[r[-1] + 1] is None:
                ks.add(r[-1])
            if not ks:
                ks.add(r[len(r) // 2])
            for i in r:
                if i in ks:
                    continue
                pk = max(k for k in ks if k < i) if any(k < i for k in ks) else None
                nk = min(k for k in ks if k > i) if any(k > i for k in ks) else None
                g = _geom(sm[i]); box = roi_box(g); bx0, by0, bx1, by1 = box
                ell = _frame_mask(frames[i], g, W, H, ax=1.0, ay=1.25, dy=0.2, fe=0.1)[by0:by1, bx0:bx1, 0]
                o_roi = cv2.GaussianBlur(frames[i][by0:by1, bx0:bx1], (0, 0), 2).astype(np.float32)
                e = 0.0
                for k in (pk, nk):
                    if k is None:
                        continue
                    mx, my = fmaps(i, k, box)
                    w_ = cv2.remap(frames[k][by0:by1, bx0:bx1], mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                    dif = np.abs(cv2.GaussianBlur(w_, (0, 0), 2).astype(np.float32) - o_roi).mean(2)
                    e = max(e, float((dif * ell).sum() / (ell.sum() + 1e-6)))
                errs[i] = e
                if e > prop_thr:
                    ks.add(i)
            keys += sorted(ks)
    for r in ([] if prop else runs):
        ks = set(r[0::key_step]) | {r[-1]}
        for i in r:
            if i in ks:
                continue
            g = _geom(sm[i]); box = roi_box(g); bx0, by0, bx1, by1 = box
            ell = _frame_mask(frames[i], g, W, H, **TIGHT)[by0:by1, bx0:bx1, 0]
            o_roi = cv2.GaussianBlur(frames[i][by0:by1, bx0:bx1], (0, 0), 2).astype(np.float32)
            e = 0.0
            for nb in (i - 1, i + 1):
                if nb not in frames:
                    e = 99.0
                    continue
                w_ = _aligned(frames[i], frames[nb], frames[nb], box)
                dif = np.abs(cv2.GaussianBlur(w_, (0, 0), 2).astype(np.float32) - o_roi).mean(2)
                e = max(e, float((dif * ell).sum() / (ell.sum() + 1e-6)))
            errs[i] = e
            if e > promote_thr:
                ks.add(i)
        keys += sorted(ks)
    if reuse_raw:
        have = sorted(k for k in (int(x[:4]) for x in os.listdir(f"{d}/{reuse_raw}") if x.endswith(".png")) if k in frames and sm[k] is not None)
        keys = [k for k in have if k in set(keys)] if prop else have
    keyset = set(keys)
    per = 14.0 * ((gen_res or res) / 896) ** 2
    est = (len(keys) * per + max(1, min(GPU_PARALLEL, -(-len(keys) // 26))) * 60) * GPU_USD_PER_SEC * EST_FACTOR
    ev = np.array(list(errs.values())) if errs else np.array([0.0])
    info = {"frames_range": [start_frame, end], "head_frames": len(todo), "key_frames": len(keys), "promoted": int(sum(1 for e in errs.values() if e > promote_thr)),
            "interp_err_pctl_50_75_90_max": [round(float(np.percentile(ev, p)), 2) for p in (50, 75, 90)] + [round(float(ev.max()), 2)],
            "est_cost_usd": round(est, 2)}
    if dry_run or (est > max_cost_usd and not reuse_raw):
        return info | {"aborted": not dry_run, "reason": None if dry_run else f"estimated ${est:.2f} > cap ${max_cost_usd}"}

    sp = f"{d}/source_{uuid.uuid4().hex[:6]}.jpg"
    _download(source_face_url, sp)
    src = cv2.imread(sp)
    fs = _fa().get(src)
    if not fs:
        raise RuntimeError("no face in source photo")
    f = max(fs, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    x0, y0, x1, y1 = f.bbox
    head_crop, _ = _crop(src, (x0 + x1) / 2, (y0 + y1) / 2 - 0.2 * (y1 - y0), 2.3 * max(y1 - y0, (x1 - x0) * 1.15), href)
    head_png = cv2.imencode(".png", head_crop)[1].tobytes()

    geoms, crops = {}, []
    for i in keys:
        g = _geom(sm[i])
        crop, M = _crop(frames[i], g[0], g[1], g[2], res)
        geoms[i] = (g, M, crop)
        crops.append((i, cv2.imencode(".png", crop)[1].tobytes()))

    outs, tg = {}, time.time()
    if reuse_raw:
        tag = reuse_raw.replace("raw_", "")[:6] + "r" + uuid.uuid4().hex[:3]
        for i in keys:
            outs[i] = cv2.imread(f"{d}/{reuse_raw}/{i:04d}.png")
        gpu_wall = 0.0
    elif seq_ref:
        chains, cur_c = [], []
        for (i, png) in crops:
            if cur_c and (i != cur_c[-1][0] + 1 or len(cur_c) >= seq_ref):
                chains.append(cur_c)
                cur_c = []
            cur_c.append((i, png))
        if cur_c:
            chains.append(cur_c)
        for rs in Worker().process_seq.map(chains, kwargs={"head_png": head_png, "steps": steps, "sampler": sampler, "seed": seed, "res": res, "href": href}):
            for idx, png in rs:
                outs[idx] = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)
        gpu_wall = time.time() - tg
        tag = uuid.uuid4().hex[:6]
        os.makedirs(f"{d}/raw_{tag}", exist_ok=True)
        for i, o in outs.items():
            cv2.imwrite(f"{d}/raw_{tag}/{i:04d}.png", o)
        jobs_vol.commit()
    else:
        n_gpu = max(1, min(GPU_PARALLEL, -(-len(crops) // 26)))   # ~26 frames per GPU container
        size = max(1, min(100, -(-len(crops) // n_gpu)))
        batches = [crops[k:k + size] for k in range(0, len(crops), size)]
        for rs in Worker().process.map(batches, kwargs={"head_png": head_png, "steps": steps, "sampler": sampler, "seed": seed,
                                                        "res": gen_res or res, "href": href, "vae_up": vae_up, "out_res": res}):
            for idx, png in rs:
                outs[idx] = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)
        gpu_wall = time.time() - tg
        tag = uuid.uuid4().hex[:6]
        os.makedirs(f"{d}/raw_{tag}", exist_ok=True)
        for i, o in outs.items():
            cv2.imwrite(f"{d}/raw_{tag}/{i:04d}.png", o)
        jobs_vol.commit()

    if gpu_only:
        return info | {"raw_dir": f"raw_{tag}", "gpu_wall_s": round(gpu_wall, 1), "gpu_frames": len(outs),
                       "frames_total": n, "fps": fps, "end": end}

    # remove frame-to-frame jitter of the generated head: the model places/scales the face slightly differently on every
    # frame; shift/scale each generated head so its deviation follows a smoothed trajectory
    align_stats = {}
    if align_fix:
        fa2 = _fa()
        D = {}
        for i in keys:
            g_, M_, crop_ = geoms[i]
            cx_, cy_, S_, w_, h_, bx, by = g_
            sc_ = res / S_
            rb = meta["track"][i]["box"] or sm[i]          # the actor's RAW per-frame head box (keeps real fast motion)
            rbx, rby, rh = (rb[0] + rb[2]) / 2, (rb[1] + rb[3]) / 2, rb[3] - rb[1]
            ex, ey = res / 2 + sc_ * (rbx - cx_), res / 2 + sc_ * (rby - cy_)
            h_ = rh
            fs_ = fa2.get(outs[i])
            if not fs_:
                continue
            ff = min(fs_, key=lambda q: abs((q.bbox[0] + q.bbox[2]) / 2 - ex) + abs((q.bbox[1] + q.bbox[3]) / 2 - ey))
            gx, gy = (ff.bbox[0] + ff.bbox[2]) / 2, (ff.bbox[1] + ff.bbox[3]) / 2
            gh = ff.bbox[3] - ff.bbox[1]
            D[i] = (gx - ex, gy - ey, float(np.log(max(gh, 1.0) / (h_ * sc_))), gx, gy)
        corr = []
        for r in runs:
            ks_ = [k for k in r if k in D]
            for i in ks_:
                ws = np.array([np.exp(-0.5 * ((k - i) / align_sigma) ** 2) for k in ks_])
                arr = np.array([D[k][:3] for k in ks_])
                sm_ = (ws[:, None] * arr).sum(0) / ws.sum()
                dx, dy, dl = np.array(D[i][:3]) - sm_
                lim = 0.06 * res
                dxc, dyc = float(np.clip(dx, -lim, lim)), float(np.clip(dy, -lim, lim))
                f = float(np.exp(-np.clip(dl, -0.06, 0.06)))
                gx, gy = D[i][3], D[i][4]
                A = np.array([[f, 0, gx - f * gx - dxc], [0, f, gy - f * gy - dyc]], np.float32)
                outs[i] = cv2.warpAffine(outs[i], A, (res, res), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
                corr.append((dxc / (res / geoms[i][0][2]), dyc / (res / geoms[i][0][2]), dl))
        if corr:
            ca = np.array(corr)
            align_stats = {"align_keys": len(corr), "corr_rms_px": round(float(np.sqrt((ca[:, 0] ** 2 + ca[:, 1] ** 2).mean())), 2),
                           "corr_scale_rms_pct": round(float(np.sqrt((ca[:, 2] ** 2).mean()) * 100), 2)}

    # composite key frames, restricted to the head (tight ellipse x pixels the model really changed)
    Kc, CM = {}, {}
    k7 = np.ones((11, 11), np.uint8)
    if seg_mask:
        sess = _seg_session(seg_full)
        GW, MW, MASKDBG, SIG = {}, {}, {}, {}
        CMP, RAWK = {}, {}
        for i in keys:
            g, M, crop = geoms[i]
            cx_, cy_, S_, w_, h_, bx, by = g
            sc_ = res / S_
            rb = meta["track"][i]["box"] or sm[i]
            fx, fy = res / 2 + sc_ * ((rb[0] + rb[2]) / 2 - cx_), res / 2 + sc_ * ((rb[1] + rb[3]) / 2 - cy_)
            capm = _head_mask(res, S_, w_, h_, bx, by, cx_, cy_, ax=1.0, ay=1.25, dy=0.2, fe=0.0) > 0.5
            mo = _head_region(sess, crop, fx, fy, w_ * sc_, h_ * sc_, capm)
            mg = _head_region(sess, outs[i], fx, fy, w_ * sc_, h_ * sc_, capm)
            u = np.maximum(mo, mg)
            kd = int(0.010 * res) | 1
            u = cv2.dilate(u, np.ones((kd, kd), np.uint8))
            mask = cv2.GaussianBlur(u, (0, 0), 0.006 * res)
            kr = int(0.06 * res) | 1
            ring = cv2.dilate(u, np.ones((kr, kr), np.uint8))
            if cm_smooth > 0:
                # colour transfer parameters are smoothed over time first (no brightness / tint "breathing")
                CMP[i] = _cm_params(outs[i], crop, ring)
                RAWK[i] = (mask, M)
                continue
            gen = _color_match(outs[i], crop, ring)
            if soft_match:
                lo = cv2.Laplacian(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), cv2.CV_32F)[mo > 0.6]
                gg_ = cv2.cvtColor(gen, cv2.COLOR_BGR2GRAY)
                lv_o = float(lo.var()) + 1e-6
                best_s, best_e = 0.0, 1e9
                for sg in np.linspace(0, 1.6, 9):
                    gb = cv2.GaussianBlur(gg_, (0, 0), sg) if sg > 0 else gg_
                    lv = float(cv2.Laplacian(gb, cv2.CV_32F)[mg > 0.6].var()) + 1e-6
                    e = abs(np.log(lv / lv_o))
                    if e < best_e:
                        best_s, best_e = float(sg), e
                SIG[i] = best_s
            inv = cv2.invertAffineTransform(M)
            GW[i] = cv2.warpAffine(gen, inv, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            MW[i] = cv2.warpAffine(mask, inv, (W, H), flags=cv2.INTER_LINEAR, borderValue=0).astype(np.float32)
        if cm_smooth > 0:
            rad_ = int(3 * cm_smooth)
            for r in runs:
                kr_ = [k for k in r if k in RAWK]
                for q_, k in enumerate(kr_):
                    acc_, ws_ = np.zeros((3, 2), np.float32), 0.0
                    for q2 in range(max(0, q_ - rad_), min(len(kr_), q_ + rad_ + 1)):
                        p2 = CMP[kr_[q2]]
                        if p2 is None:
                            continue
                        w2 = float(np.exp(-0.5 * ((q2 - q_) / cm_smooth) ** 2))
                        acc_ += p2 * w2
                        ws_ += w2
                    gen = _cm_apply(outs[k], acc_ / ws_ if ws_ > 0 else None)
                    mask, M = RAWK[k]
                    inv = cv2.invertAffineTransform(M)
                    GW[k] = cv2.warpAffine(gen, inv, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
                    MW[k] = cv2.warpAffine(mask, inv, (W, H), flags=cv2.INTER_LINEAR, borderValue=0).astype(np.float32)
            del RAWK
        sig_s = {}
        for r in runs:
            kr_ = [k for k in r if k in SIG]
            for q_, k in enumerate(kr_):
                win_ = [SIG[x] for x in kr_[max(0, q_ - 4):q_ + 5]]
                sig_s[k] = float(np.median(win_)) if win_ else 0.0
        def compose(i):
            nb = [k for k in (i - 1, i + 1) if k in MW and k in keyset]
            m = MW[i] * 0.5 + sum(MW[k] * (0.5 / len(nb)) for k in nb) if nb else MW[i]
            m = np.clip(m, 0, 1).astype(np.float32)
            CM[i] = m
            gwi = cv2.GaussianBlur(GW[i], (0, 0), sig_s[i]) if (soft_match and sig_s.get(i, 0) > 0.2) else GW[i]
            Kc[i] = (frames[i].astype(np.float32) * (1 - m[..., None]) + gwi.astype(np.float32) * m[..., None]).clip(0, 255).astype(np.uint8)

        for i in keys:
            compose(i)

        # head stabiliser: every generated head is registered (similarity transform from dense flow inside the head
        # mask) onto the motion-compensated average of its neighbours, so its own frame-to-frame shake is removed
        # while the real head motion (carried by the neighbours' flow alignment) stays
        kruns = [[k for k in r if k in keyset] for r in runs]
        krun_of = {k: kr for kr in kruns for k in kr}
        stab_stats = []
        for _it in range(stab_iters):
            corr = {}
            for i in keys:
                kr_ = krun_of[i]
                nbs = [k for k in kr_ if k != i and abs(k - i) <= stab_taps]
                if not nbs:
                    continue
                g = _geom(sm[i])
                x0, y0, x1, y1 = roi_box(g)
                tgt = np.mean([_aligned(frames[i], frames[k], Kc[k], (x0, y0, x1, y1)).astype(np.float32) for k in nbs], 0)
                sc = min(1.0, 480.0 / max(x1 - x0, y1 - y0))
                gc = cv2.cvtColor(cv2.resize(Kc[i][y0:y1, x0:x1], None, fx=sc, fy=sc), cv2.COLOR_BGR2GRAY)
                gt = cv2.cvtColor(cv2.resize(tgt.clip(0, 255).astype(np.uint8), (gc.shape[1], gc.shape[0])), cv2.COLOR_BGR2GRAY)
                fl = cv2.calcOpticalFlowFarneback(gc, gt, None, 0.5, 3, 15, 3, 5, 1.1, 0)
                mm = cv2.resize(CM[i][y0:y1, x0:x1], (gc.shape[1], gc.shape[0]))
                ys, xs = np.mgrid[0:gc.shape[0]:3, 0:gc.shape[1]:3]
                sel = mm[ys, xs] > 0.6
                if sel.sum() < 40:
                    continue
                src_p = np.stack([xs[sel], ys[sel]], 1).astype(np.float32)
                dst_p = src_p + fl[ys[sel], xs[sel]]
                A, _ = cv2.estimateAffinePartial2D(src_p, dst_p, method=cv2.RANSAC, ransacReprojThreshold=1.5)
                if A is None:
                    continue
                s_ = float(np.hypot(A[0, 0], A[1, 0]))
                th = float(np.arctan2(A[1, 0], A[0, 0]))
                tx, ty = float(A[0, 2]) / sc, float(A[1, 2]) / sc
                # damp + clamp (never more than 3 % shift / 4 % scale / 3 degrees per frame)
                lim = 0.03 * max(x1 - x0, y1 - y0)
                s_ = 1 + stab_w * float(np.clip(s_ - 1, -0.04, 0.04))
                th = stab_w * float(np.clip(th, -0.052, 0.052))
                L = np.array([[s_ * np.cos(th), -s_ * np.sin(th)], [s_ * np.sin(th), s_ * np.cos(th)]], np.float32)
                t = np.array([tx, ty], np.float32) * stab_w
                nt = float(np.hypot(*t))
                if nt > lim:
                    t *= lim / nt
                # the transform was estimated in ROI coordinates (origin x0,y0): convert to full-frame coordinates
                o = np.array([x0, y0], np.float32)
                corr[i] = np.hstack([L, (t + o - L @ o)[:, None]]).astype(np.float32)
                stab_stats.append((float(np.hypot(*t)), abs(s_ - 1)))
            for i, A in corr.items():
                GW[i] = cv2.warpAffine(GW[i], A, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
                MW[i] = cv2.warpAffine(MW[i], A, (W, H), flags=cv2.INTER_LINEAR, borderValue=0)
            for i in keys:
                compose(i)
        if stab_stats:
            ss = np.array(stab_stats)
            align_stats = {"stab_corr_rms_px": round(float(np.sqrt((ss[:, 0] ** 2).mean())), 2),
                           "stab_scale_rms_pct": round(float(np.sqrt((ss[:, 1] ** 2).mean())) * 100, 2)}
        del GW, MW
    else:
        for i in keys:
            g, M, crop = geoms[i]
            cx_, cy_, S_, w_, h_, bx, by = g
            mask = _head_mask(res, S_, w_, h_, bx, by, cx_, cy_, **TIGHT)
            gen = _color_match(outs[i], crop, _head_mask(res, S_, w_, h_, bx, by, cx_, cy_))
            inv = cv2.invertAffineTransform(M)
            gw = cv2.warpAffine(gen, inv, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            mw = cv2.warpAffine(mask, inv, (W, H), flags=cv2.INTER_LINEAR, borderValue=0)
            diff = cv2.GaussianBlur(np.abs(gw.astype(np.float32) - frames[i].astype(np.float32)).mean(2), (0, 0), 3)
            cm = np.clip((diff - 6) / 16, 0, 1)
            cm = cv2.GaussianBlur(cv2.dilate(cm, k7), (0, 0), 4)
            m = (np.maximum(cm, 0.6 * mw) * mw).astype(np.float32)
            CM[i] = m
            Kc[i] = (frames[i].astype(np.float32) * (1 - m[..., None]) + gw.astype(np.float32) * m[..., None]).clip(0, 255).astype(np.uint8)

    def blend_roi(base_roi, aligned_list, thr=18, weights=None):
        ref = cv2.GaussianBlur(base_roi, (0, 0), 3).astype(np.float32)
        acc, ws = base_roi.astype(np.float32), 1.0
        for q, a in enumerate(aligned_list):
            diff = np.abs(cv2.GaussianBlur(a, (0, 0), 3).astype(np.float32) - ref).mean(2)
            wv = (diff < thr).astype(np.float32)[..., None] * (weights[q] if weights else 0.5)
            acc += a.astype(np.float32) * wv
            ws = ws + wv
        return acc / ws

    final = {}
    run_of = {i: r for r in runs for i in r}
    for i in todo:
        o_t = frames[i]
        g = _geom(sm[i])
        box = roi_box(g)
        bx0, by0, bx1, by1 = box
        res_f = o_t.copy()
        if i in keyset:
            ks = [k for k in run_of[i] if k in keyset]
            pos = ks.index(i)
            offs = [o for t_ in range(1, taps + 1) for o in (-t_, t_)]
            nbs = [ks[pos + o] for o in offs if 0 <= pos + o < len(ks) and abs(ks[pos + o] - i) <= 3 * abs(o)]
            wmap = {1: 0.5, 2: 0.35, 3: 0.2}
            wts = [wmap.get(abs(ks.index(k) - pos), 0.15) for k in nbs]
            al = [(_aligned(Kc[i], Kc[k], Kc[k], box) if flow_comp else _aligned(o_t, frames[k], Kc[k], box)) for k in nbs]
            m = CM[i][by0:by1, bx0:bx1, None]
            base = Kc[i][by0:by1, bx0:bx1]
            new = base.astype(np.float32) * (1 - m) + blend_roi(base, al, weights=wts) * m if al else base.astype(np.float32)
        else:
            ell = _frame_mask(o_t, g, W, H, **TIGHT)[by0:by1, bx0:bx1, 0]
            ksr = [k for k in run_of[i] if k in keyset]
            pk = max([k for k in ksr if k < i], default=None)
            nk = min([k for k in ksr if k > i], default=None)
            if pk is None or nk is None:
                pk = nk = (pk if pk is not None else nk)
            if prop:
                # carry the full head (and its mask, so hair is included) from both keys with the original motion
                wr = {}
                for k in {pk, nk}:
                    mx, my = fmaps(i, k, box)
                    wr[k] = (cv2.remap(Kc[k][by0:by1, bx0:bx1], mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).astype(np.float32),
                             cv2.remap(CM[k][by0:by1, bx0:bx1], mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0))
                wb = 0.5 if pk == nk else (i - pk) / float(nk - pk)
                (a, ma), (b, mb) = wr[pk], wr[nk]
                dab = np.abs(cv2.GaussianBlur(a, (0, 0), 3) - cv2.GaussianBlur(b, (0, 0), 3)).mean(2)[..., None]
                pick = np.where(dab < prop_gate, (1 - wb) * a + wb * b, a if wb < 0.5 else b)
                m = ((1 - wb) * ma + wb * mb)[..., None]
                new = o_t[by0:by1, bx0:bx1].astype(np.float32) * (1 - m) + pick * m
                res_f[by0:by1, bx0:bx1] = new.clip(0, 255).astype(np.uint8)
                final[i] = res_f
                continue
            a = _aligned(o_t, frames[pk], Kc[pk], box)
            b = _aligned(o_t, frames[nk], Kc[nk], box)
            m = (ell)[..., None]
            wb = 0.5 if pk == nk else (i - pk) / float(nk - pk)
            dab = np.abs(cv2.GaussianBlur(a, (0, 0), 3).astype(np.float32) - cv2.GaussianBlur(b, (0, 0), 3).astype(np.float32)).mean(2)[..., None]
            near = a.astype(np.float32) if wb < 0.5 else b.astype(np.float32)
            pick = np.where(dab < 25, (1 - wb) * a.astype(np.float32) + wb * b.astype(np.float32), near)
            new = o_t[by0:by1, bx0:bx1].astype(np.float32) * (1 - m) + pick * m
        res_f[by0:by1, bx0:bx1] = new.clip(0, 255).astype(np.uint8)
        if fix_top and by0 == 0:
            ellt = _frame_mask(o_t, g, W, H, **TIGHT)[0:60, bx0:bx1, 0]
            if ellt[0:4].max() > 0.5:
                B = 28
                refl = res_f[B:2 * B, bx0:bx1][::-1].astype(np.float32)
                wt = np.clip(ellt[0:B] * 1.0, 0, 1)[..., None] * np.linspace(1.0, 0.0, B, dtype=np.float32)[:, None, None] ** 0.5
                res_f[0:B, bx0:bx1] = (res_f[0:B, bx0:bx1].astype(np.float32) * (1 - wt) + refl * wt).clip(0, 255).astype(np.uint8)
        if grain:
            ellg = _frame_mask(o_t, g, W, H, **TIGHT)[by0:by1, bx0:bx1, 0]
            go = cv2.cvtColor(o_t[by0:by1, bx0:bx1], cv2.COLOR_BGR2GRAY).astype(np.float32)
            gg = cv2.cvtColor(res_f[by0:by1, bx0:bx1], cv2.COLOR_BGR2GRAY).astype(np.float32)
            hp_o = go - cv2.GaussianBlur(go, (0, 0), 1.5)
            hp_g = gg - cv2.GaussianBlur(gg, (0, 0), 1.5)
            ring, core = ellg < 0.03, ellg > 0.6
            if ring.sum() > 2000 and core.sum() > 2000:
                so, sg = float(hp_o[ring].std()), float(hp_g[core].std())
                sig = min(grain_max, float(np.sqrt(max(so * so - sg * sg, 0.0))))
                if sig > 0.3:
                    rng = np.random.default_rng(1234 + i)
                    nz = cv2.GaussianBlur(rng.standard_normal(gg.shape).astype(np.float32), (0, 0), 0.7) * sig * 1.6
                    roi = res_f[by0:by1, bx0:bx1].astype(np.float32) + (nz * ellg)[..., None]
                    res_f[by0:by1, bx0:bx1] = roi.clip(0, 255).astype(np.uint8)
        final[i] = res_f

    # second temporal pass over the finished frames (also covers interpolated frames): every head is averaged with its
    # motion-aligned neighbours (+-post_taps); neighbours that do not match (fast motion, occlusion) are rejected per pixel
    if post_taps:
        final2 = {}
        for i in todo:
            g = _geom(sm[i])
            box = roi_box(g)
            bx0, by0, bx1, by1 = box
            r_ = run_of[i]
            nbs = [i + o for t_ in range(1, post_taps + 1) for o in (-t_, t_) if (i + o) in final and (i + o) in r_]
            if not nbs:
                continue
            m = CM[i][by0:by1, bx0:bx1, None] if i in CM else _frame_mask(frames[i], g, W, H, **TIGHT)[by0:by1, bx0:bx1]
            wts = [post_w / abs(k - i) for k in nbs]
            al = [_aligned(frames[i], frames[k], final[k], box) for k in nbs]
            base = final[i][by0:by1, bx0:bx1]
            new = base.astype(np.float32) * (1 - m) + blend_roi(base, al, thr=post_thr, weights=wts) * m
            f2 = final[i].copy()
            f2[by0:by1, bx0:bx1] = new.clip(0, 255).astype(np.uint8)
            final2[i] = f2
        final.update(final2)

    if core:
        out_path = f"{d}/{out_name}"
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(fps), "-i", "-",
               "-c:v", "libx264", "-crf", "16", "-preset", "medium", "-pix_fmt", "yuv420p", out_path]
        p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        for i in range(core[0], core[1]):
            p.stdin.write(final.get(i, frames[i]).tobytes())
        p.stdin.close()
        p.wait()
        jobs_vol.commit()
        return {"part": out_name, "frames": core}
    out_path = f"{d}/out2_{tag}.mp4"
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(fps), "-i", "-",
           "-ss", f"{start_frame / fps:.3f}", "-t", f"{(end - start_frame) / fps:.3f}", "-i", f"{d}/video.mp4", "-map", "0:v", "-map", "1:a?",
           "-c:v", "libx264", "-crf", "16", "-preset", "medium", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k", "-shortest", out_path]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for i in range(start_frame, start_frame + len(frames)):
        p.stdin.write(final.get(i, frames[i]).tobytes())
    p.stdin.close()
    p.wait()
    jobs_vol.commit()
    uploaded = _put_file(out_path, output_put_url, "video/mp4") if output_put_url else None
    return info | {"output": out_path.replace(JOBS + "/", ""), "raw_dir": f"raw_{tag}", "gpu_wall_s": round(gpu_wall, 1),
                   "total_s": round(time.time() - t0, 1), "res": res, "uploaded": uploaded} | align_stats



@app.function(image=cpu_image, volumes={JOBS: jobs_vol}, cpu=4, memory=8192, timeout=1800)
def measure(template_id: str, names: list, offset: int = 0):
    """High-frequency wobble of the head position/scale in a video (lower = steadier). 'video.mp4' = the real actor as baseline."""
    import cv2, numpy as np
    jobs_vol.reload()
    d = f"{JOBS}/{template_id}"
    meta = json.load(open(f"{d}/track.json"))
    sm = _smooth_track(meta["track"], meta["frames"])
    fa = _fa()
    res, P = {}, {}
    for name in names:
        cap = cv2.VideoCapture(f"{d}/{name}")
        pos, i = {}, (offset if name != "video.mp4" else 0)
        while True:
            ok, fr = cap.read()
            if not ok or i >= len(sm):
                break
            if sm[i] is not None:
                b = sm[i]
                tx, ty = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
                fs = fa.get(fr)
                if fs:
                    q = min(fs, key=lambda q: abs((q.bbox[0] + q.bbox[2]) / 2 - tx) + abs((q.bbox[1] + q.bbox[3]) / 2 - ty))
                    pos[i] = ((q.bbox[0] + q.bbox[2]) / 2, (q.bbox[1] + q.bbox[3]) / 2, q.bbox[3] - q.bbox[1])
            i += 1
        cap.release()
        P[name] = pos
        idx = sorted(pos)
        k = np.exp(-0.5 * (np.arange(-6, 7) / 2.0) ** 2)
        k /= k.sum()
        rx, rs = [], []
        seg = []
        for j in idx + [None]:
            if j is not None and (not seg or j == seg[-1] + 1):
                seg.append(j)
                continue
            if len(seg) >= 9:
                a = np.array([pos[t] for t in seg], dtype=float)
                a[:, 2] = np.log(a[:, 2])
                pad = np.pad(a, ((6, 6), (0, 0)), mode="edge")
                smo = np.stack([np.convolve(pad[:, c], k, mode="valid") for c in range(3)], 1)
                r = a - smo
                rx += list(np.sqrt(r[:, 0] ** 2 + r[:, 1] ** 2))
                rs += list(r[:, 2])
            seg = [j] if j is not None else []
        res[name] = {"frames_measured": len(idx), "pos_jitter_rms_px": round(float(np.sqrt(np.mean(np.square(rx)))), 2) if rx else None,
                     "scale_jitter_rms_pct": round(float(np.sqrt(np.mean(np.square(rs))) * 100), 2) if rs else None}
    if "video.mp4" in P:
        k = np.exp(-0.5 * (np.arange(-6, 7) / 2.0) ** 2)
        k /= k.sum()
        for name in names:
            if name == "video.mp4":
                continue
            common = sorted(set(P[name]) & set(P["video.mp4"]))
            seg, add, adds = [], [], []
            for j in common + [None]:
                if j is not None and (not seg or j == seg[-1] + 1):
                    seg.append(j)
                    continue
                if len(seg) >= 9:
                    a = np.array([[P[name][t][0] - P["video.mp4"][t][0], P[name][t][1] - P["video.mp4"][t][1],
                                   np.log(P[name][t][2] / P["video.mp4"][t][2])] for t in seg])
                    pad = np.pad(a, ((6, 6), (0, 0)), mode="edge")
                    smo = np.stack([np.convolve(pad[:, c], k, mode="valid") for c in range(3)], 1)
                    r = a - smo
                    add += list(np.sqrt(r[:, 0] ** 2 + r[:, 1] ** 2))
                    adds += list(r[:, 2])
                seg = [j] if j is not None else []
            res[name]["ADDED_pos_jitter_rms_px"] = round(float(np.sqrt(np.mean(np.square(add)))), 2) if add else None
            res[name]["ADDED_scale_jitter_rms_pct"] = round(float(np.sqrt(np.mean(np.square(adds))) * 100), 2) if adds else None
    return res



@app.function(image=cpu_image, volumes={JOBS: jobs_vol}, cpu=4, memory=24576, timeout=3600)
def refine_video(template_id: str, video_name: str, denoise: float = 0.3, start_frame: int = 0, max_frames: int = 0,
                 out_name: str = "", crop_res: int = 704, chunk: int = 81, overlap: int = 16, steps: int = 20, cfg: float = 5.0,
                 seed: int = 42, prompt: str = WAN_PROMPT, max_cost_usd: float = 3.0, dry_run: bool = False,
                 crop_scale: float = 1.0):
    """Temporal refine of a finished swap (video_name inside the template folder) with the Wan video model.
    Only the head + hair (face-parsing mask) of the refined clip is pasted back, colour-matched to the input."""
    import cv2, numpy as np
    t0 = time.time()
    jobs_vol.reload()
    d = f"{JOBS}/{template_id}"
    meta = json.load(open(f"{d}/track.json"))
    n, fps, W, H = meta["frames"], meta["fps"], meta["w"], meta["h"]
    end = n if not max_frames else min(n, start_frame + max_frames)
    sm = _smooth_track(meta["track"], n)
    runs, cur = [], []
    for i in range(start_frame, end):
        if sm[i] is None:
            continue
        if cur and i != cur[-1] + 1:
            runs.append(cur)
            cur = []
        cur.append(i)
    if cur:
        runs.append(cur)
    runs = [r for r in runs if len(r) >= 5]
    # chunks of <= `chunk` frames with `overlap` frames shared between neighbours; padded to 4k+1 frames for the VAE
    chunks = []
    for r in runs:
        s = 0
        while True:
            seg = r[s:s + chunk]
            chunks.append((seg, s > 0, s + chunk < len(r)))
            if s + chunk >= len(r):
                break
            s += chunk - overlap
    padded = sum(((len(c[0]) - 1 + 3) // 4) * 4 + 1 for c in chunks)
    est = (padded * 0.9 * (crop_res / 704) ** 2 * (denoise * steps / 6) + len(chunks) * 15 + 120) * GPU_USD_PER_SEC * EST_FACTOR
    info = {"frames_range": [start_frame, end], "head_frames": sum(len(r) for r in runs), "chunks": len(chunks),
            "est_cost_usd": round(est, 2)}
    if dry_run or est > max_cost_usd:
        return info | {"aborted": not dry_run}

    cap = cv2.VideoCapture(f"{d}/{video_name}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frames = {}
    for i in range(start_frame, end):
        ok, fr = cap.read()
        if not ok:
            break
        frames[i] = fr
    cap.release()
    geo, crops = {}, {}
    for r in runs:
        for i in r:
            g = _geom_clamped(sm[i], W, H)
            g = (g[0], g[1], g[2] * crop_scale) + tuple(g[3:])   # tighter crop = more head pixels for the video model
            geo[i] = g
            crops[i] = _crop(frames[i], g[0], g[1], g[2], crop_res)

    def clip_bytes(seg):
        L2 = ((len(seg) - 1 + 3) // 4) * 4 + 1
        idx = seg + [seg[-1]] * (L2 - len(seg))
        tmp = f"/tmp/clip_{uuid.uuid4().hex[:6]}.mp4"
        p = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{crop_res}x{crop_res}",
                              "-r", str(fps), "-i", "-", "-c:v", "libx264", "-crf", "8", "-preset", "medium", "-pix_fmt", "yuv444p", tmp],
                             stdin=subprocess.PIPE)
        for i in idx:
            p.stdin.write(crops[i][0].tobytes())
        p.stdin.close()
        p.wait()
        b = open(tmp, "rb").read()
        os.remove(tmp)
        return b

    tg = time.time()
    clips = [clip_bytes(c[0]) for c in chunks]
    outs = list(RefineWorker().refine.map(clips, kwargs={"denoise": denoise, "steps": steps, "cfg": cfg, "seed": seed, "prompt": prompt}))
    gpu_wall = time.time() - tg

    acc, wsum = {}, {}
    for (seg, has_prev, has_next), pngs in zip(chunks, outs):
        L = len(seg)
        for pos, i in enumerate(seg):
            w = 1.0
            if has_prev:
                w = min(w, (pos + 1) / (overlap + 1))
            if has_next:
                w = min(w, (L - pos) / (overlap + 1))
            im = cv2.imdecode(np.frombuffer(pngs[pos], np.uint8), cv2.IMREAD_COLOR).astype(np.float32)
            if im.shape[0] != crop_res:
                im = cv2.resize(im, (crop_res, crop_res), interpolation=cv2.INTER_CUBIC)
            acc[i] = acc.get(i, 0) + im * w
            wsum[i] = wsum.get(i, 0) + w

    sess = _seg_session(False)
    final = {}
    for i in acc:
        ref = (acc[i] / wsum[i]).clip(0, 255).astype(np.uint8)
        crop, M = crops[i]
        cx_, cy_, S_, w_, h_, bx, by = geo[i]
        sc = crop_res / S_
        capm = _head_mask(crop_res, S_, w_, h_, bx, by, cx_, cy_, ax=1.0, ay=1.25, dy=0.2, fe=0.0) > 0.5
        fx, fy = crop_res / 2 + sc * (bx - cx_), crop_res / 2 + sc * (by - cy_)
        m = _head_region(sess, crop, fx, fy, w_ * sc, h_ * sc, capm)
        m = cv2.GaussianBlur(cv2.dilate(m, np.ones((int(0.012 * crop_res) | 1,) * 2, np.uint8)), (0, 0), 0.008 * crop_res)
        e_ = np.clip(np.minimum(np.arange(crop_res), np.arange(crop_res)[::-1]) / (0.05 * crop_res), 0, 1).astype(np.float32)
        m = m * np.minimum(e_[:, None], e_[None, :])          # fade out towards the crop border (tight crops)
        ring =cv2.dilate(m, np.ones((int(0.06 * crop_res) | 1,) * 2, np.uint8))
        ref = _color_match(ref, crop, ring)
        inv = cv2.invertAffineTransform(M)
        gw = cv2.warpAffine(ref, inv, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE).astype(np.float32)
        mw = cv2.warpAffine(m, inv, (W, H), flags=cv2.INTER_LINEAR, borderValue=0)[..., None]
        final[i] = (frames[i].astype(np.float32) * (1 - mw) + gw * mw).clip(0, 255).astype(np.uint8)

    out_name = out_name or f"refined_{uuid.uuid4().hex[:6]}.mp4"
    out_path = f"{d}/{out_name}"
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(fps), "-i", "-",
           "-ss", f"{start_frame / fps:.3f}", "-t", f"{(end - start_frame) / fps:.3f}", "-i", f"{d}/video.mp4", "-map", "0:v", "-map", "1:a?",
           "-c:v", "libx264", "-crf", "16", "-preset", "medium", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k", "-shortest", out_path]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for i in range(start_frame, start_frame + len(frames)):
        p.stdin.write(final.get(i, frames[i]).tobytes())
    p.stdin.close()
    p.wait()
    jobs_vol.commit()
    return info | {"output": out_name, "gpu_wall_s": round(gpu_wall, 1), "total_s": round(time.time() - t0, 1), "denoise": denoise}


@app.function(image=cpu_image, volumes={JOBS: jobs_vol}, cpu=8, memory=32768, timeout=5400)
def animate_video(template_id: str, bfs_video: str, start_frame: int = 0, max_frames: int = 0, out_name: str = "",
                  crop_res: int = 832, crop_scale: float = 0.8, chunk: int = 77, overlap: int = 12, steps: int = 6, seed: int = 42,
                  prompt: str = ANIM_PROMPT, face_src: str = "bfs", pose_src: str = "orig"):
    """Video-native head replacement with Wan 2.2 Animate, guided by a finished BFS swap (bfs_video, same template):
    one BFS frame is the identity reference, the original footage drives pose + expression. Output = original video
    with the generated head pasted back (face-parsing mask, colour-matched)."""
    import cv2, numpy as np
    t0 = time.time()
    jobs_vol.reload()
    d = f"{JOBS}/{template_id}"
    meta = json.load(open(f"{d}/track.json"))
    n, fps, W, H = meta["frames"], meta["fps"], meta["w"], meta["h"]
    end = n if not max_frames else min(n, start_frame + max_frames)
    sm = _smooth_track(meta["track"], n)

    def read(name):
        cap = cv2.VideoCapture(f"{d}/{name}")
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        fr = {}
        for i in range(start_frame, end):
            ok, f = cap.read()
            if not ok:
                break
            fr[i] = f
        cap.release()
        return fr

    orig, bfs = read("video.mp4"), read(bfs_video)
    runs, cur = [], []
    for i in range(start_frame, end):
        if sm[i] is None or i not in orig:
            continue
        if cur and i != cur[-1] + 1:
            runs.append(cur)
            cur = []
        cur.append(i)
    if cur:
        runs.append(cur)
    runs = [r for r in runs if len(r) >= 5]
    sess = _seg_session(False)
    fa = _fa()
    R = crop_res
    geo, oc, bc, hm = {}, {}, {}, {}
    for r in runs:
        for i in r:
            g = _geom_clamped(sm[i], W, H)
            g = (g[0], g[1], g[2] * crop_scale) + tuple(g[3:])
            geo[i] = g
            oc[i] = _crop(orig[i], g[0], g[1], g[2], R)
            bc[i] = _crop(bfs[i], g[0], g[1], g[2], R)[0]
            cx_, cy_, S_, w_, h_, bx, by = g
            sc = R / S_
            capm = _head_mask(R, S_, w_, h_, bx, by, cx_, cy_, ax=1.0, ay=1.25, dy=0.2, fe=0.0) > 0.5
            fx, fy = R / 2 + sc * (bx - cx_), R / 2 + sc * (by - cy_)
            mo = _head_region(sess, oc[i][0], fx, fy, w_ * sc, h_ * sc, capm)
            mb = _head_region(sess, bc[i], fx, fy, w_ * sc, h_ * sc, capm)
            hm[i] = np.maximum(mo, mb)

    # identity reference: the most frontal, confidently detected BFS head in the range
    best, ref_i = -9.0, None
    for r in runs:
        for i in r[::3]:
            fl = fa.get(bc[i])
            if not fl:
                continue
            f = max(fl, key=lambda q: (q.bbox[2] - q.bbox[0]) * (q.bbox[3] - q.bbox[1]))
            k = f.kps
            ed = float(np.hypot(*(k[1] - k[0]))) + 1e-3
            s = float(f.det_score) - abs(float(k[2][0] - (k[0][0] + k[1][0]) / 2)) / ed + 0.3 * min(1.0, ed / (0.12 * R))
            if s > best:
                best, ref_i = s, i
    if ref_i is None:
        raise RuntimeError("no usable reference frame")
    ref_png = cv2.imencode(".png", bc[ref_i])[1].tobytes()

    def enc(imgs, size, gray=False):
        tmp = f"/tmp/v_{uuid.uuid4().hex[:6]}.mp4"
        p = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{size}x{size}", "-r", str(fps),
                              "-i", "-", "-c:v", "libx264", "-crf", "6", "-preset", "medium", "-pix_fmt", "yuv444p", tmp], stdin=subprocess.PIPE)
        for im in imgs:
            p.stdin.write(im.tobytes())
        p.stdin.close()
        p.wait()
        b = open(tmp, "rb").read()
        os.remove(tmp)
        return b

    jobs = []
    for r in runs:
        s = 0
        while True:
            seg = r[s:s + chunk]
            jobs.append((seg, s > 0, s + chunk < len(r)))
            if s + chunk >= len(r):
                break
            s += chunk - overlap
    payloads = []
    for seg, _, _ in jobs:
        L2 = ((len(seg) - 1 + 3) // 4) * 4 + 1
        idx = seg + [seg[-1]] * (L2 - len(seg))
        kd = int(0.03 * R) | 1
        bg, msk, face, og = [], [], [], []
        for i in idx:
            m = cv2.dilate(hm[i], np.ones((kd, kd), np.uint8)) > 0.4
            b = oc[i][0].copy()
            b[m] = 0
            bg.append(b)
            msk.append(np.repeat((m * 255).astype(np.uint8)[..., None], 3, 2))
            og.append(bc[i] if pose_src == "bfs" else oc[i][0])
            rb = meta["track"][i]["box"] or sm[i]
            fcx, fcy, fs_ = (rb[0] + rb[2]) / 2, (rb[1] + rb[3]) / 2, 1.4 * max(rb[2] - rb[0], rb[3] - rb[1])
            # expressions: from the BFS swap by default - the face crops leak identity into the result, so they
            # must already show the new person (the original performer's crops made the head look like him)
            face.append(_crop((bfs if face_src == "bfs" else orig)[i], fcx, fcy, fs_, 512)[0])
        payloads.append(({"ref.png": ref_png, "bg.mp4": enc(bg, R), "mask.mp4": enc(msk, R), "orig.mp4": enc(og, R),
                          "face.mp4": enc(face, 512)}, L2))
    tg = time.time()
    calls = [AnimateWorker().animate.spawn(pl, R, R, L2, seed, steps, prompt) for pl, L2 in payloads]
    outs = [c.get() for c in calls]
    gpu_wall = time.time() - tg

    acc, wsum = {}, {}
    for (seg, has_prev, has_next), pngs in zip(jobs, outs):
        L = len(seg)
        for pos, i in enumerate(seg):
            if pos >= len(pngs):
                break
            w = 1.0
            if has_prev:
                w = min(w, (pos + 1) / (overlap + 1))
            if has_next:
                w = min(w, (L - pos) / (overlap + 1))
            im = cv2.imdecode(np.frombuffer(pngs[pos], np.uint8), cv2.IMREAD_COLOR).astype(np.float32)
            if im.shape[0] != R:
                im = cv2.resize(im, (R, R), interpolation=cv2.INTER_CUBIC)
            acc[i] = acc.get(i, 0) + im * w
            wsum[i] = wsum.get(i, 0) + w

    final = {}
    for i in acc:
        gen = (acc[i] / wsum[i]).clip(0, 255).astype(np.uint8)
        crop, M = oc[i]
        cx_, cy_, S_, w_, h_, bx, by = geo[i]
        sc = R / S_
        capm = _head_mask(R, S_, w_, h_, bx, by, cx_, cy_, ax=1.0, ay=1.25, dy=0.2, fe=0.0) > 0.5
        fx, fy = R / 2 + sc * (bx - cx_), R / 2 + sc * (by - cy_)
        mg = _head_region(sess, gen, fx, fy, w_ * sc, h_ * sc, capm)
        u = np.maximum(hm[i], mg)
        u = cv2.dilate(u, np.ones((int(0.012 * R) | 1,) * 2, np.uint8))
        m = cv2.GaussianBlur(u, (0, 0), 0.007 * R)
        e_ = np.clip(np.minimum(np.arange(R), np.arange(R)[::-1]) / (0.05 * R), 0, 1).astype(np.float32)
        m = m * np.minimum(e_[:, None], e_[None, :])
        ring = cv2.dilate(u, np.ones((int(0.06 * R) | 1,) * 2, np.uint8))
        gen = _color_match(gen, crop, ring)
        inv = cv2.invertAffineTransform(M)
        gw = cv2.warpAffine(gen, inv, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE).astype(np.float32)
        mw = cv2.warpAffine(m, inv, (W, H), flags=cv2.INTER_LINEAR, borderValue=0)[..., None]
        final[i] = (orig[i].astype(np.float32) * (1 - mw) + gw * mw).clip(0, 255).astype(np.uint8)

    out_name = out_name or f"animate_{uuid.uuid4().hex[:6]}.mp4"
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(fps), "-i", "-",
           "-ss", f"{start_frame / fps:.3f}", "-t", f"{(end - start_frame) / fps:.3f}", "-i", f"{d}/video.mp4", "-map", "0:v", "-map", "1:a?",
           "-c:v", "libx264", "-crf", "14", "-preset", "medium", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k", "-shortest", f"{d}/{out_name}"]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for i in range(start_frame, start_frame + len(orig)):
        p.stdin.write(final.get(i, orig[i]).tobytes())
    p.stdin.close()
    p.wait()
    jobs_vol.commit()
    return {"output": out_name, "chunks": len(jobs), "ref_frame": ref_i, "head_frames": len(final), "gpu_wall_s": round(gpu_wall, 1),
            "total_s": round(time.time() - t0, 1)}


@app.function(image=cpu_image, volumes={JOBS: jobs_vol}, cpu=8, memory=32768, timeout=5400)
def userflow_video(template_id: str, source_face_url: str, start_frame: int = 0, max_frames: int = 0, out_name: str = "",
                   crop_res: int = 1280, seed: int = 42, prompt: str = USER_PROMPT, grow: int = 40, blur: int = 30):
    """Runs the user-supplied ComfyUI workflow (UserFlowWorker) on every tracked head frame of a template video:
    head crop at crop_res (their workflow scales the target to 1280), face+hair mask grown/blurred like their
    MaskGrow(40, 30), result (2x from SeedVR2) scaled back and pasted with the same mask. No temporal smoothing,
    so the output shows the workflow as it is."""
    import cv2, numpy as np
    t0 = time.time()
    jobs_vol.reload()
    d = f"{JOBS}/{template_id}"
    meta = json.load(open(f"{d}/track.json"))
    n, fps, W, H = meta["frames"], meta["fps"], meta["w"], meta["h"]
    end = n if not max_frames else min(n, start_frame + max_frames)
    sm = _smooth_track(meta["track"], n)
    cap = cv2.VideoCapture(f"{d}/video.mp4")
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frames = {}
    for i in range(start_frame, end):
        ok, f = cap.read()
        if not ok:
            break
        frames[i] = f
    cap.release()
    todo = [i for i in frames if sm[i] is not None]
    fa = _fa()
    sp = f"/tmp/src_{uuid.uuid4().hex[:6]}.jpg"
    _download(source_face_url, sp)
    src = cv2.imread(sp)
    fs = fa.get(src)
    if not fs:
        raise RuntimeError("no face in source photo")
    f0 = max(fs, key=lambda q: (q.bbox[2] - q.bbox[0]) * (q.bbox[3] - q.bbox[1]))
    x0, y0, x1, y1 = f0.bbox
    src_crop, _ = _crop(src, (x0 + x1) / 2, (y0 + y1) / 2 - 0.15 * (y1 - y0), 2.5 * max(y1 - y0, x1 - x0), 1024)
    source_png = cv2.imencode(".png", src_crop)[1].tobytes()
    sess = _seg_session(False)
    R = crop_res
    geo, crops, masks, batch = {}, {}, {}, []
    for i in todo:
        g = _geom_clamped(sm[i], W, H)
        geo[i] = g
        crop, M = _crop(frames[i], g[0], g[1], g[2], R)
        crops[i] = (crop, M)
        cx_, cy_, S_, w_, h_, bx, by = g
        sc = R / S_
        capm = _head_mask(R, S_, w_, h_, bx, by, cx_, cy_, ax=1.0, ay=1.25, dy=0.2, fe=0.0) > 0.5
        fx, fy = R / 2 + sc * (bx - cx_), R / 2 + sc * (by - cy_)
        m = _head_region(sess, crop, fx, fy, w_ * sc, h_ * sc, capm)
        m = cv2.dilate((m > 0.5).astype(np.float32), np.ones((2 * grow + 1, 2 * grow + 1), np.uint8))
        m = cv2.GaussianBlur(m, (0, 0), blur / 2.0)
        masks[i] = m
        batch.append((i, cv2.imencode(".png", crop)[1].tobytes(),
                      cv2.imencode(".png", np.repeat((m * 255).astype(np.uint8)[..., None], 3, 2))[1].tobytes()))
    n_gpu = max(1, min(GPU_PARALLEL, -(-len(batch) // 10)))
    size = -(-len(batch) // n_gpu)
    tg = time.time()
    outs = {}
    for rs in UserFlowWorker().process.map([batch[k:k + size] for k in range(0, len(batch), size)],
                                           kwargs={"source_png": source_png, "seed": seed, "prompt": prompt, "up_res": 2 * R}):
        for idx, png in rs:
            outs[idx] = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)
    gpu_wall = time.time() - tg
    final = {}
    for i, o in outs.items():
        o = cv2.resize(o, (R, R), interpolation=cv2.INTER_AREA)
        crop, M = crops[i]
        inv = cv2.invertAffineTransform(M)
        gw = cv2.warpAffine(o, inv, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE).astype(np.float32)
        mw = cv2.warpAffine(masks[i], inv, (W, H), flags=cv2.INTER_LINEAR, borderValue=0)[..., None]
        final[i] = (frames[i].astype(np.float32) * (1 - mw) + gw * mw).clip(0, 255).astype(np.uint8)
    out_name = out_name or f"userflow_{uuid.uuid4().hex[:6]}.mp4"
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(fps), "-i", "-",
           "-ss", f"{start_frame / fps:.3f}", "-t", f"{(end - start_frame) / fps:.3f}", "-i", f"{d}/video.mp4", "-map", "0:v", "-map", "1:a?",
           "-c:v", "libx264", "-crf", "14", "-preset", "medium", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k", "-shortest", f"{d}/{out_name}"]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for i in range(start_frame, start_frame + len(frames)):
        p.stdin.write(final.get(i, frames[i]).tobytes())
    p.stdin.close()
    p.wait()
    jobs_vol.commit()
    return {"output": out_name, "head_frames": len(final), "gpus": n_gpu, "gpu_wall_s": round(gpu_wall, 1), "total_s": round(time.time() - t0, 1)}


ARCFACE_128 = [[0.36167656, 0.40387734], [0.63696719, 0.40235469], [0.50019687, 0.56044219], [0.38710391, 0.72160547], [0.61507734, 0.72034453]]
FACE_ONLY_CLASSES = [1, 2, 4, 5, 6, 7, 10, 11, 12]     # skin, nose, eyes, brows, mouth, lips (no hair / ears / neck)


@app.function(image=cpu_image, volumes={JOBS: jobs_vol}, cpu=8, memory=16384, timeout=3600)
def hyperswap_video(template_id: str, video_name: str, source_face_url: str, start_frame: int = 0, max_frames: int = 0,
                    out_name: str = "", model: str = "hyperswap_1a_256", pixel_boost: int = 768, kps_sigma: float = 1.5):
    """EXPERIMENTAL face-only swap with FaceFusion HyperSwap (256 px ONNX, licence: ResearchRAIL - research use).
    video_name = 'video.mp4' swaps the original footage; a finished BFS output gives the BFS + HyperSwap hybrid.
    The model is deterministic (same input -> same face), 5-point landmarks are smoothed over time against wobble,
    'pixel_boost' runs the 256 px model on (pixel_boost/256)^2 interleaved tiles for more detail on large faces, and the
    face-parsing mask keeps hands / food in front of the face."""
    import cv2, numpy as np, onnxruntime as ort, requests
    t0 = time.time()
    jobs_vol.reload()
    mp = f"{JOBS}/_models/{model}.onnx"
    if not os.path.exists(mp):
        os.makedirs(os.path.dirname(mp), exist_ok=True)
        r = requests.get(f"https://huggingface.co/facefusion/models-3.3.0/resolve/main/{model}.onnx", timeout=600)
        r.raise_for_status()
        open(mp, "wb").write(r.content)
        jobs_vol.commit()
    so = ort.SessionOptions()
    so.intra_op_num_threads = 8
    swapper = ort.InferenceSession(mp, so, providers=["CPUExecutionProvider"])
    in_names = [x.name for x in swapper.get_inputs()]

    d = f"{JOBS}/{template_id}"
    meta = json.load(open(f"{d}/track.json"))
    n, fps, W, H = meta["frames"], meta["fps"], meta["w"], meta["h"]
    end = n if not max_frames else min(n, start_frame + max_frames)
    sm = _smooth_track(meta["track"], n)
    fa = _fa()
    sp = f"/tmp/src_{uuid.uuid4().hex[:6]}.jpg"
    _download(source_face_url, sp)
    fs = fa.get(cv2.imread(sp))
    if not fs:
        raise RuntimeError("no face in source photo")
    src_emb = max(fs, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])).normed_embedding.reshape(1, -1).astype(np.float32)

    cap = cv2.VideoCapture(f"{d}/{video_name}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frames = {}
    for i in range(start_frame, end):
        ok, fr = cap.read()
        if not ok:
            break
        frames[i] = fr
    cap.release()
    kps = {}
    for i, fr in frames.items():
        if sm[i] is None:
            continue
        b = sm[i]
        tx, ty = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        fl = fa.get(fr)
        if fl:
            q = min(fl, key=lambda q: abs((q.bbox[0] + q.bbox[2]) / 2 - tx) + abs((q.bbox[1] + q.bbox[3]) / 2 - ty))
            if abs((q.bbox[0] + q.bbox[2]) / 2 - tx) < (b[2] - b[0]):
                kps[i] = q.kps.astype(np.float32)
    # temporal smoothing of the landmarks within contiguous runs (removes alignment wobble)
    idx = sorted(kps)
    runs, cur = [], []
    for i in idx:
        if cur and i != cur[-1] + 1:
            runs.append(cur)
            cur = []
        cur.append(i)
    if cur:
        runs.append(cur)
    skps = {}
    rad = int(3 * kps_sigma)
    for r in runs:
        arr = np.stack([kps[i] for i in r])
        k = np.exp(-0.5 * (np.arange(-rad, rad + 1) / max(kps_sigma, 1e-3)) ** 2)
        k /= k.sum()
        pad = np.pad(arr, ((rad, rad), (0, 0), (0, 0)), mode="edge")
        for q, i in enumerate(r):
            skps[i] = (pad[q:q + 2 * rad + 1] * k[:, None, None]).sum(0) if kps_sigma > 0 else arr[q]

    sess = _seg_session(False)
    pb, t = pixel_boost, pixel_boost // 256
    tmpl = np.array(ARCFACE_128, np.float32) * pb
    out = {}
    for i in skps:
        fr = frames[i]
        M = cv2.estimateAffinePartial2D(skps[i], tmpl, method=cv2.RANSAC, ransacReprojThreshold=100)[0]
        crop = cv2.warpAffine(fr, M, (pb, pb), borderMode=cv2.BORDER_REPLICATE, flags=cv2.INTER_AREA)
        tiles = crop.reshape(256, t, 256, t, 3).transpose(1, 3, 0, 2, 4).reshape(t * t, 256, 256, 3)
        outs_ = []
        for tile in tiles:
            x = ((tile[..., ::-1] / 255.0 - 0.5) / 0.5).transpose(2, 0, 1)[None].astype(np.float32)
            feed = {nm: (src_emb if nm == "source" else x) for nm in in_names}
            y = swapper.run(None, feed)[0][0].transpose(1, 2, 0) * 0.5 + 0.5
            outs_.append(y.clip(0, 1)[..., ::-1] * 255)
        sw = np.stack(outs_).reshape(t, t, 256, 256, 3).transpose(2, 0, 3, 1, 4).reshape(pb, pb, 3).astype(np.float32)
        lab = cv2.resize(_parse_labels(sess, crop).astype(np.uint8), (pb, pb), interpolation=cv2.INTER_NEAREST)
        m = np.isin(lab, FACE_ONLY_CLASSES).astype(np.float32)
        m = cv2.erode(m, np.ones((int(0.01 * pb) | 1,) * 2, np.uint8))
        e_ = np.clip(np.minimum(np.arange(pb), np.arange(pb)[::-1]) / (0.08 * pb), 0, 1).astype(np.float32)
        m = cv2.GaussianBlur(m, (0, 0), 0.012 * pb) * np.minimum(e_[:, None], e_[None, :])
        inv = cv2.invertAffineTransform(M)
        gw = cv2.warpAffine(sw, inv, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        mw = cv2.warpAffine(m, inv, (W, H), flags=cv2.INTER_LINEAR, borderValue=0)[..., None]
        out[i] = (fr.astype(np.float32) * (1 - mw) + gw * mw).clip(0, 255).astype(np.uint8)

    out_name = out_name or f"hyper_{uuid.uuid4().hex[:6]}.mp4"
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(fps), "-i", "-",
                          "-c:v", "libx264", "-crf", "12", "-preset", "medium", "-pix_fmt", "yuv420p", f"{d}/{out_name}"], stdin=subprocess.PIPE)
    for i in range(start_frame, start_frame + len(frames)):
        p.stdin.write(out.get(i, frames[i]).tobytes())
    p.stdin.close()
    p.wait()
    jobs_vol.commit()
    return {"output": out_name, "frames": len(frames), "swapped": len(out), "inputs": in_names,
            "outputs": [x.name for x in swapper.get_outputs()], "total_s": round(time.time() - t0, 1)}


def _put_file(path, url, ctype):
    import requests
    with open(path, "rb") as f:
        r = requests.put(url, data=f, headers={"Content-Type": ctype}, timeout=600)
    return {"status": r.status_code, "ok": r.ok}


def _geom_clamped(box, W, H):
    g = _geom_base(box)
    cx_, cy_, S_ = g[0], g[1], g[2]
    if S_ <= W:
        cx_ = min(max(cx_, S_ / 2), W - S_ / 2)
    if S_ <= H:
        cy_ = min(max(cy_, S_ / 2), H - S_ / 2)
    return (cx_, cy_, S_) + tuple(g[3:])


@app.function(image=cpu_image, volumes={JOBS: jobs_vol}, cpu=4, memory=16384, timeout=1800)
def swap_image(image_url: str, source_face_url: str, target_face_url: str = "", face_index: int = -1, quality: str = "best",
               seed: int = 42, steps: int = 8, sampler: str = "euler", output_put_url: str = "", max_cost_usd: float = 0.5,
               skin_match: float = 0.9, keep_neck: bool = False, protect_cloth: bool = True,
               match_sharpness: bool = True, best_of: int = 1, relight: float = 0.0, jaw_blend: bool = True,
               harmonize: float = 0.0, cfg: float = 1.0, distill: bool = True, neck_fade: float = 0.22,
               bg_match: bool = True, neck_depth: float = 0.76, engine: str = "bfs", prompt: str = "", lora: float = 1.0,
               tone_finish: bool = False):
    """Head swap on ONE image. image_url: the picture to edit; source_face_url: the head to put in.
    Pick the person to replace with target_face_url (a face crop of them), face_index (0 = left-most) or by default the largest face."""
    import cv2, numpy as np
    t0 = time.time()
    jobs_vol.reload()
    res, href = PRESETS.get(quality, PRESETS["best"])["res"], PRESETS.get(quality, PRESETS["best"])["href"]
    os.makedirs(f"{JOBS}/images", exist_ok=True)
    jid = uuid.uuid4().hex[:10]
    if image_url.startswith("volume:test/") and ".." not in image_url:   # internal test shortcut, confined to test/
        img = cv2.imread(f"{JOBS}/{image_url[7:]}")
    else:
        _download(image_url, f"{JOBS}/images/{jid}_in")
        img = cv2.imread(f"{JOBS}/images/{jid}_in")
    if img is None:
        raise RuntimeError("could not read image_url")
    H, W = img.shape[:2]
    fa = _fa()
    faces = fa.get(img)
    if not faces:
        raise RuntimeError("no face found in image_url")
    if target_face_url:
        _download(target_face_url, f"{JOBS}/images/{jid}_tf")
        temb = _target_embedding(fa, cv2.imread(f"{JOBS}/images/{jid}_tf"))
        f = max(faces, key=lambda q: float(np.dot(q.normed_embedding, temb))) if temb is not None else max(faces, key=lambda q: (q.bbox[2] - q.bbox[0]) * (q.bbox[3] - q.bbox[1]))
    elif face_index >= 0:
        f = sorted(faces, key=lambda q: q.bbox[0])[min(face_index, len(faces) - 1)]
    else:
        f = max(faces, key=lambda q: (q.bbox[2] - q.bbox[0]) * (q.bbox[3] - q.bbox[1]))
    box = [float(x) for x in f.bbox]

    # head reference
    _download(source_face_url, f"{JOBS}/images/{jid}_src")
    src = cv2.imread(f"{JOBS}/images/{jid}_src")
    sf = fa.get(src)
    if not sf:
        raise RuntimeError("no face in source_face_url")
    q = max(sf, key=lambda q: (q.bbox[2] - q.bbox[0]) * (q.bbox[3] - q.bbox[1]))
    x0, y0, x1, y1 = q.bbox
    head_crop, _ = _crop(src, (x0 + x1) / 2, (y0 + y1) / 2 - 0.2 * (y1 - y0), 2.3 * max(y1 - y0, (x1 - x0) * 1.15), href)
    head_png = cv2.imencode(".png", head_crop)[1].tobytes()

    g = _geom_clamped(box, W, H)
    cx_, cy_, S_, w_, h_, bx, by = g
    if PRESETS.get(quality, PRESETS["best"]).get("adaptive"):
        res = _res_for_size(S_, res)
    crop, M = _crop(img, cx_, cy_, S_, res)
    sc_ = res / S_
    est = ((res / 1024) ** 2 * 12 + 100) * EST_FACTOR
    if est * GPU_USD_PER_SEC > max_cost_usd:
        raise RuntimeError("estimated cost above max_cost_usd")
    sess = _seg_session(False)
    fx, fy = res / 2 + sc_ * (bx - cx_), res / 2 + sc_ * (by - cy_)
    crop_png = cv2.imencode(".png", crop)[1].tobytes()
    wk = Worker()
    cands = []
    for k in range(max(1, int(best_of))):
        sd = seed + 1000 * k
        if engine == "klein":
            out = KleinWorker().process.remote([(0, crop_png)], head_png, seed=sd, steps=(steps if steps <= 12 else 4),
                                               prompt=prompt, lora=lora, res=res)
        else:
            out = wk.process.remote([(0, crop_png)], head_png, steps=steps, sampler=sampler, seed=sd, res=res, href=href,
                                    cfg=cfg, distill=distill, engine=engine, prompt=prompt, lora=lora)
        cands.append((sd, cv2.imdecode(np.frombuffer(out[0][1], np.uint8), cv2.IMREAD_COLOR)))
    scores = []
    if len(cands) > 1:
        # pick the best: likeness to the new face, same head pose/expression as the original, no extra hair (bun / tuft)
        near = lambda fs: min(fs, key=lambda z: (z.bbox[0] + z.bbox[2] - 2 * fx) ** 2 + (z.bbox[1] + z.bbox[3] - 2 * fy) ** 2)
        f68 = _fa68()
        t68 = f68.get(crop)
        t68 = near(t68) if t68 else None
        lo_h = cv2.resize(_parse_labels(sess, crop).astype(np.uint8), (res, res), interpolation=cv2.INTER_NEAREST)
        hrow = np.nonzero(np.isin(lo_h, HEAD_CLASSES).any(1))[0]
        top = int(hrow.min()) if len(hrow) else 0
        for sd, gimg in cands:
            gf = fa.get(gimg)
            if not gf:
                scores.append((-9.0, sd, {}))
                continue
            ident = float(np.dot(near(gf).normed_embedding, q.normed_embedding))
            expr, pose = 0.0, 0.0
            g68 = f68.get(gimg)
            if t68 is not None and g68:
                expr, pose = _expr_pose_diff(t68, near(g68))
            lg_h = cv2.resize(_parse_labels(sess, gimg).astype(np.uint8), (res, res), interpolation=cv2.INTER_NEAREST)
            fa_ = float(max(1, (w_ * sc_) ** 2))
            extra = float((lg_h[: max(0, top - int(0.03 * res))] == 13).sum()) / fa_
            # open mouth / teeth (label 10 = inside of the mouth) should match the original
            mouth = abs(float((lg_h == 10).sum()) - float((lo_h == 10).sum())) / fa_
            sc = ident - 2.0 * expr - 0.012 * pose - 2.0 * extra - 8.0 * mouth
            scores.append((sc, sd, {"identity": round(ident, 3), "expression_diff": round(expr, 3), "pose_diff_deg": round(pose, 1),
                                    "extra_hair": round(extra, 3), "mouth_diff": round(mouth, 4)}))
        best = max(range(len(cands)), key=lambda i: scores[i][0])
        seed, gen = cands[best]
    else:
        seed, gen = cands[0]
    capm = _head_mask(res, S_, w_, h_, bx, by, cx_, cy_, ax=1.0, ay=1.25, dy=0.2, fe=0.0) > 0.5
    hc = [c for c in HEAD_CLASSES if c != 17] if keep_neck else None   # keep_neck: paste face/hair/ears only, original neck stays
    mo = _head_region(sess, crop, fx, fy, w_ * sc_, h_ * sc_, capm, hc)
    mg = _head_region(sess, gen, fx, fy, w_ * sc_, h_ * sc_, capm, hc)
    # all of the ORIGINAL hair that is connected to the head (ponytail, hair behind the neck/ears) is replaced too,
    # otherwise a cut-off piece of the old hair stays next to the new jaw
    lo_hr = cv2.resize(_parse_labels(sess, crop).astype(np.uint8), (res, res), interpolation=cv2.INTER_NEAREST)
    hair_o = (lo_hr == 13).astype(np.uint8)
    hair_o[int(min(res, fy + 1.4 * h_ * sc_)):] = 0
    nlab, cc_h = cv2.connectedComponents(hair_o, connectivity=8)
    touch = cv2.dilate((mo > 0.5).astype(np.uint8), np.ones((int(0.02 * res) | 1,) * 2, np.uint8)) > 0
    keep_ids = np.unique(cc_h[touch & (cc_h > 0)])
    old_hair_px = 0
    oh = np.zeros((res, res), np.float32)
    if len(keep_ids):
        oh = np.isin(cc_h, keep_ids).astype(np.float32)
        oh = cv2.dilate(oh, np.ones((int(0.008 * res) | 1,) * 2, np.uint8))
        old_hair_px = int(oh.sum())
        mo = np.maximum(mo, cv2.GaussianBlur(oh, (0, 0), 1.5))
    u = np.maximum(mo, mg)
    kd = int(0.010 * res) | 1
    u = cv2.dilate(u, np.ones((kd, kd), np.uint8))
    mask = cv2.GaussianBlur(u, (0, 0), 0.006 * res)
    prot_px = 0
    cloth_keep = None
    if protect_cloth:
        # clothes and necklaces of the original are never replaced (only new hair may fall over them)
        lo_ = cv2.resize(_parse_labels(sess, crop).astype(np.uint8), (res, res), interpolation=cv2.INTER_NEAREST)
        lg_ = cv2.resize(_parse_labels(sess, gen).astype(np.uint8), (res, res), interpolation=cv2.INTER_NEAREST)
        keep_ = np.isin(lo_, [16, 18]) & (lg_ != 13)
        # below the chin, everything in the original that is not skin-coloured (necklace, collar, cloth, background) stays;
        # skin colour model = Mahalanobis distance in LAB to the original face skin
        L_ = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).astype(np.float32)
        em_ = np.zeros((res, res), np.uint8)
        cv2.ellipse(em_, (int(fx), int(fy + 0.05 * h_ * sc_)), (max(2, int(0.28 * w_ * sc_)), max(2, int(0.30 * h_ * sc_))), 0, 0, 360, 1, -1)
        s_ = L_[em_ > 0]
        if len(s_) > 200:
            d_ = L_ - s_.mean(0)
            ci_ = np.linalg.inv(np.cov(s_.T) + np.eye(3) * 4.0)
            ns_ = np.sqrt(np.einsum("...i,ij,...j->...", d_, ci_, d_)) > 3.0
            ns_ &= (lo_ != 13) & (lg_ != 13)
            ns_[: int(fy + 0.58 * h_ * sc_)] = False          # only below the chin
            ko_, kc_ = int(0.006 * res) | 1, int(0.012 * res) | 1
            ns_ = cv2.morphologyEx(ns_.astype(np.uint8), cv2.MORPH_OPEN, np.ones((ko_, ko_), np.uint8))
            ns_ = cv2.morphologyEx(ns_, cv2.MORPH_CLOSE, np.ones((kc_, kc_), np.uint8))
            # per column: from the first non-skin pixel (top of necklace / collar) downwards everything stays original,
            # so the new neck only fills chin -> necklace/collar and never redraws jewellery, collars or clothes
            ns_ = np.maximum.accumulate(ns_, axis=0) > 0
            keep_ |= ns_ & (lg_ != 13)
        keep_ &= oh < 0.5                                   # old hair (ponytail) is never "protected"
        prot_px = int((keep_ & (mask > 0.1)).sum())
        cloth_keep = keep_
        mask = mask * (1.0 - cv2.GaussianBlur(keep_.astype(np.float32), (0, 0), 0.003 * res))
        # the new neck is used just under the jaw and fades back to the original neck lower down, so necklaces / collars
        # that look like skin are never redrawn (new hair may still fall lower)
        yy_ = np.arange(res, dtype=np.float32)[:, None]
        ya_, yb_ = fy + (float(neck_depth) - 0.18) * h_ * sc_, fy + float(neck_depth) * h_ * sc_
        fade_ = np.clip((yb_ - yy_) / max(1.0, yb_ - ya_), 0.0, 1.0)
        mask = mask * np.maximum(np.maximum(fade_, cv2.GaussianBlur((lg_ == 13).astype(np.float32), (0, 0), 0.004 * res)),
                                 cv2.GaussianBlur(oh, (0, 0), 0.004 * res))
    neck_px = 0
    if keep_neck and neck_fade > 0:
        # soft neck transition: no hard line at the jaw - the new skin/stubble fades into the original neck over a short
        # band, only where BOTH images show neck/skin (never over clothes, jewellery or background)
        lo_n = cv2.resize(_parse_labels(sess, crop).astype(np.uint8), (res, res), interpolation=cv2.INTER_NEAREST)
        lg_n = cv2.resize(_parse_labels(sess, gen).astype(np.uint8), (res, res), interpolation=cv2.INTER_NEAREST)
        yy_n = np.arange(res, dtype=np.float32)[:, None]
        y_top, y_bot = fy + 0.42 * h_ * sc_, fy + (0.42 + float(neck_fade)) * h_ * sc_
        ramp = np.clip((y_bot - yy_n) / max(1.0, y_bot - y_top), 0.0, 1.0) * (yy_n >= y_top - 0.08 * h_ * sc_)
        neckpix = (np.isin(lg_n, [1, 17]) & np.isin(lo_n, [1, 17])).astype(np.float32)
        neckpix = cv2.erode(neckpix, np.ones((int(0.012 * res) | 1,) * 2, np.uint8))
        ext = cv2.GaussianBlur(neckpix * ramp, (0, 0), 0.008 * res)
        if cloth_keep is not None:
            ext *= 1.0 - cloth_keep.astype(np.float32)
        neck_px = int((ext > 0.2).sum())
        mask = np.maximum(mask, ext)
    kr = int(0.06 * res) | 1
    gen = _color_match(gen, crop, cv2.dilate(u, np.ones((kr, kr), np.uint8)))
    if skin_match > 0:
        gen = _skin_tone_match(gen, crop, sess, mask, strength=float(skin_match))
    if relight > 0:
        lo_r = cv2.resize(_parse_labels(sess, crop).astype(np.uint8), (res, res), interpolation=cv2.INTER_NEAREST)
        lg_r = cv2.resize(_parse_labels(sess, gen).astype(np.uint8), (res, res), interpolation=cv2.INTER_NEAREST)
        gen = _relight(gen, crop, lo_r, lg_r, w_ * sc_, alpha=float(relight))
    bg_px = 0
    if bg_match:
        lo_b = cv2.resize(_parse_labels(sess, crop).astype(np.uint8), (res, res), interpolation=cv2.INTER_NEAREST)
        lg_b = cv2.resize(_parse_labels(sess, gen).astype(np.uint8), (res, res), interpolation=cv2.INTER_NEAREST)
        gen, bg_px = _bg_match(gen, crop, lo_b, lg_b, mask)
    jaw_px = 0
    if keep_neck and jaw_blend:
        lo_j = cv2.resize(_parse_labels(sess, crop).astype(np.uint8), (res, res), interpolation=cv2.INTER_NEAREST)
        lg_j = cv2.resize(_parse_labels(sess, gen).astype(np.uint8), (res, res), interpolation=cv2.INTER_NEAREST)
        gen, jaw_px = _jaw_blend(gen, crop, lo_j, lg_j, mask, fy, w_ * sc_)
    inv = cv2.invertAffineTransform(M)
    gw = cv2.warpAffine(gen, inv, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT_101)
    mw = cv2.warpAffine(mask, inv, (W, H), flags=cv2.INTER_LINEAR, borderValue=0)[..., None].astype(np.float32)
    sharp_info = {}
    if match_sharpness:
        gw, sharp_info = _match_sharpness_grain(img, gw, mw[..., 0], seed)
    final = (img.astype(np.float32) * (1 - mw) + gw.astype(np.float32) * mw).clip(0, 255).astype(np.uint8)
    tone_info = {}
    if tone_finish:
        final, tone_info = _finish_tone(img, final, mw[..., 0], box)
        # everything outside the pasted head/neck is restored bit-exact from the original
        keep_out = cv2.dilate((mw[..., 0] > 0.002).astype(np.uint8), np.ones((5, 5), np.uint8)) == 0
        final[keep_out] = img[keep_out]
    harm_info = {}
    if harmonize > 0:
        # harmonize pass: the finished crop goes through the base model at low strength, so light, edges, skin texture and
        # the jaw-neck join are re-drawn by the model itself; only the head + a thin seam band is taken from it
        cropF, _ = _crop(final, cx_, cy_, S_, res)
        hp = wk.refine.remote(cv2.imencode(".png", cropF)[1].tobytes(), seed=seed, steps=8, denoise=float(harmonize), res=res)
        hz = cv2.imdecode(np.frombuffer(hp, np.uint8), cv2.IMREAD_COLOR)
        if hz.shape[:2] != (res, res):
            hz = cv2.resize(hz, (res, res), interpolation=cv2.INTER_LANCZOS4)
        kh = int(0.035 * res) | 1
        hm = cv2.dilate((mask > 0.3).astype(np.uint8), np.ones((kh, kh), np.uint8)).astype(np.float32)
        if cloth_keep is not None:
            hm *= 1.0 - cloth_keep.astype(np.float32)
        hm = cv2.GaussianBlur(hm, (0, 0), 0.010 * res)
        try:
            hf = fa.get(hz)
            if hf:
                h0 = min(hf, key=lambda z: (z.bbox[0] + z.bbox[2] - 2 * fx) ** 2 + (z.bbox[1] + z.bbox[3] - 2 * fy) ** 2)
                gf0 = fa.get(gen)
                g0 = min(gf0, key=lambda z: (z.bbox[0] + z.bbox[2] - 2 * fx) ** 2 + (z.bbox[1] + z.bbox[3] - 2 * fy) ** 2) if gf0 else None
                harm_info = {"identity_after": round(float(np.dot(h0.normed_embedding, q.normed_embedding)), 3),
                             "identity_before": round(float(np.dot(g0.normed_embedding, q.normed_embedding)), 3) if g0 is not None else None}
        except Exception as e:
            harm_info = {"id_check_error": str(e)[:100]}
        hzw = cv2.warpAffine(hz, inv, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT_101)
        hmw = cv2.warpAffine(hm, inv, (W, H), flags=cv2.INTER_LINEAR, borderValue=0)[..., None]
        final = (final.astype(np.float32) * (1 - hmw) + hzw.astype(np.float32) * hmw).clip(0, 255).astype(np.uint8)
    outp = f"{JOBS}/images/{jid}.png"
    cv2.imwrite(outp, final)
    jobs_vol.commit()
    uploaded = _put_file(outp, output_put_url, "image/png") if output_put_url else None
    return {"output": f"images/{jid}.png", "faces_found": len(faces), "size": [W, H], "res": res, "seconds": round(time.time() - t0, 1), "uploaded": uploaded,
            "model": "Qwen-Image-2.1 + bfs_head_v1.1_qwen_2.1", "code": "v15-studio-klein", "engine": engine, "tone": tone_info, "jaw_blend_px": int(jaw_px), "neck_fade_px": int(neck_px),
            "old_hair_px": int(old_hair_px), "bg_match_px": int(bg_px), "harmonize": harm_info,
            "protected_px": int(prot_px), "seed_used": seed, "candidates": [s[2] | {"seed": s[1], "score": round(s[0], 3)} for s in scores],
            "sharpness": sharp_info}


@app.function(image=cpu_image, volumes={JOBS: jobs_vol}, cpu=1, memory=2048, timeout=5400)
def swap_video(source_face_url: str, template_id: str = "", video_url: str = "", target_face_url: str = "", quality: str = "best",
               max_cost_usd: float = 5.0, output_put_url: str = ""):
    """One call: preprocess (if needed) + head swap. Pass template_id of an already-preprocessed video, or video_url + target_face_url."""
    jobs_vol.reload()
    tid = template_id or uuid.uuid4().hex
    if not os.path.exists(f"{JOBS}/{tid}/track.json"):
        if not (video_url and target_face_url):
            raise RuntimeError("template not preprocessed: pass video_url and target_face_url")
        preprocess.remote(tid, video_url, target_face_url)
    r = swap_job.remote(tid, source_face_url, quality=quality, max_cost_usd=max_cost_usd, output_put_url=output_put_url)
    return r | {"template_id": tid, "quality": quality}
