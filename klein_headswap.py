"""Head swap on Flux 2 Klein 4B + BFS head LoRA (Alissonerdx/BFS-Best-Face-Swap), hosted on Modal.

Model:   black-forest-labs/FLUX.2-klein-4B (Apache 2.0, step-distilled, 4 steps)
LoRA:    bfs_head_v1_flux-klein_4b.safetensors, strength 1.0 by default (author's recommendation)
Inputs:  Picture 1 = BODY / target image, Picture 2 = HEAD / reference face. Never reverse them.

Deploy:  modal deploy klein_headswap.py
Test:    modal run klein_headswap.py --body body.jpg --head face.jpg --out result.png

HTTP API (Bearer token from the Modal secret "klein-api-token", key API_TOKEN):
  GET  /           small upload page
  POST /headswap   multipart: body=<file>, head=<file>, optional lora_strength, steps, seed, extra_prompt,
                   max_side, lora (v1 | v1.1),
                   mode (crop = paste head onto full-res original [default] | full)  -> image/png
"""
import io
import os

import modal

APP_NAME = "headswap-klein"
MODEL_ID = "black-forest-labs/FLUX.2-klein-4B"
LORA_REPO = "Alissonerdx/BFS-Best-Face-Swap"
LORAS = {
    "v1": "bfs_head_v1_flux-klein_4b.safetensors",
    "v1.1": "bfs_head_v1.1_optional_flux-klein_4b.safetensors",
}
# Official prompt from docs/flux-2-klein.md in the BFS repo.
BFS_PROMPT = (
    "head_swap: start with Picture 1 as the base image, keeping its lighting, environment, and background. "
    "remove the head from Picture 1 completely and replace it with the head from Picture 2, strictly preserving "
    "the hair, eye color, and nose structure of Picture 2. copy the direction of the eye, head rotation, and "
    "micro-expressions from Picture 1. Keep the body, clothing, pose, framing, and background intact."
)
HF_CACHE = "/hf-cache"

app = modal.App(APP_NAME)
cache_vol = modal.Volume.from_name("klein-hf-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "curl")
    .pip_install("torch==2.8.0", "torchvision", index_url="https://download.pytorch.org/whl/cu128")
    .pip_install(
        "git+https://github.com/huggingface/diffusers.git",
        "transformers>=4.57", "accelerate", "peft>=0.17", "safetensors", "sentencepiece",
        "huggingface_hub", "pillow", "numpy", "opencv-python-headless", "fastapi[standard]", "python-multipart",
    )
    .run_commands("mkdir -p /det && curl -sL -o /det/yunet.onnx "
                  "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx && test $(stat -c%s /det/yunet.onnx) -gt 100000")
    .env({"HF_HOME": HF_CACHE})
)
YUNET = "/det/yunet.onnx"


def _fit_size(w, h, max_side):
    """Keep the aspect ratio, scale the long side to max_side, round to multiples of 16."""
    s = max_side / max(w, h)
    return max(256, int(w * s) // 16 * 16), max(256, int(h * s) // 16 * 16)


def _largest_face(img):
    """(x, y, w, h) of the largest face found by OpenCV YuNet, or None."""
    import cv2
    import numpy as np

    a = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    s = min(1.0, 1600 / max(a.shape[:2]))                     # detect on a <=1600px copy
    small = cv2.resize(a, None, fx=s, fy=s) if s < 1 else a
    det = cv2.FaceDetectorYN.create(YUNET, "", (small.shape[1], small.shape[0]), 0.6)
    _, faces = det.detect(small)
    if faces is None or len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])[:4] / s
    return float(x), float(y), float(w), float(h)


def _head_crop_box(face, W, H, scale=3.2):
    """Square crop around the face with room for hair, neck and shoulders (face sits in the upper middle)."""
    x, y, w, h = face
    side = min(max(w, h) * scale, W, H)
    cx, cy = x + w / 2, y + h * 0.75
    left = int(min(max(cx - side / 2, 0), W - side))
    top = int(min(max(cy - side / 2, 0), H - side))
    return left, top, left + int(side), top + int(side)


def _paste_mask(orig_crop, gen_crop, face_in_crop):
    """Feathered mask: pixels the model changed, limited to a generous ellipse around head + neck."""
    import cv2
    import numpy as np

    o = np.asarray(orig_crop).astype(np.float32)
    g = np.asarray(gen_crop).astype(np.float32)
    n = o.shape[0]
    x, y, w, h = face_in_crop
    ell = np.zeros(o.shape[:2], np.uint8)
    cv2.ellipse(ell, (int(x + w / 2), int(y + h * 0.55)), (int(w * 1.05), int(h * 1.25)), 0, 0, 360, 255, -1)
    cv2.rectangle(ell, (int(x + w * 0.15), int(y + h * 0.9)), (int(x + w * 0.85), int(y + h * 1.6)), 255, -1)  # neck
    diff = cv2.GaussianBlur(np.abs(g - o).mean(2), (0, 0), n / 200)
    changed = (diff > 18).astype(np.uint8) * 255
    changed = cv2.dilate(changed, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (n // 25 | 1,) * 2))
    face_core = np.zeros_like(ell)
    cv2.ellipse(face_core, (int(x + w / 2), int(y + h / 2)), (int(w * 0.6), int(h * 0.75)), 0, 0, 360, 255, -1)
    m = np.maximum(cv2.bitwise_and(changed, ell), face_core).astype(np.float32) / 255
    m = cv2.GaussianBlur(m, (0, 0), n / 60)
    return np.clip(m * 1.15, 0, 1)[..., None]


@app.cls(image=image, gpu="L40S", volumes={HF_CACHE: cache_vol}, timeout=900, scaledown_window=120, max_containers=4)
class KleinHeadSwap:
    @modal.enter()
    def load(self):
        import torch
        from diffusers import Flux2KleinPipeline

        self.torch = torch
        self.pipe = Flux2KleinPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16).to("cuda")
        for name, fname in LORAS.items():
            self.pipe.load_lora_weights(LORA_REPO, weight_name=fname, adapter_name=name.replace(".", "_"))
        cache_vol.commit()

    def _run(self, body, head, prompt, steps, seed, max_side):
        w, h = _fit_size(*body.size, max_side)
        return self.pipe(
            image=[body, head],          # Picture 1 = body, Picture 2 = head (never reverse)
            prompt=prompt,
            width=w, height=h,
            num_inference_steps=int(steps),
            guidance_scale=1.0,
            generator=self.torch.Generator("cuda").manual_seed(int(seed)),
        ).images[0]

    @modal.method()
    def swap(self, body_png: bytes, head_png: bytes, lora_strength: float = 1.0, steps: int = 4, seed: int = 42,
             extra_prompt: str = "", max_side: int = 1024, lora: str = "v1", mode: str = "crop") -> bytes:
        """mode="crop": swap on a head crop and paste back onto the full-res original (keeps text, logos, detail).
        mode="full": regenerate the whole image at max_side (what the plain ComfyUI workflow does)."""
        import numpy as np
        from PIL import Image, ImageOps

        if lora not in LORAS:
            raise ValueError(f"lora must be one of {list(LORAS)}")
        body = ImageOps.exif_transpose(Image.open(io.BytesIO(body_png))).convert("RGB")
        head = ImageOps.exif_transpose(Image.open(io.BytesIO(head_png))).convert("RGB")
        self.pipe.set_adapters([lora.replace(".", "_")], adapter_weights=[float(lora_strength)])
        prompt = BFS_PROMPT + (" " + extra_prompt.strip() if extra_prompt.strip() else "")

        face = _largest_face(body) if mode == "crop" else None
        if face is None:
            out = self._run(body, head, prompt, steps, seed, max_side)
        else:
            box = _head_crop_box(face, *body.size)
            crop = body.crop(box)
            gen = self._run(crop, head, prompt, steps, seed, max_side).resize(crop.size, Image.LANCZOS)
            fx, fy, fw, fh = face
            m = _paste_mask(crop, gen, (fx - box[0], fy - box[1], fw, fh))
            merged = np.asarray(gen, np.float32) * m + np.asarray(crop, np.float32) * (1 - m)
            out = body.copy()
            out.paste(Image.fromarray(merged.round().clip(0, 255).astype(np.uint8)), box[:2])
        buf = io.BytesIO()
        out.save(buf, format="PNG")
        return buf.getvalue()


PAGE = """<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Klein Head Swap</title><style>body{font-family:system-ui,sans-serif;max-width:720px;margin:24px auto;padding:0 16px}
label{display:block;margin:10px 0 4px;font-weight:600}input,button{font-size:15px}img{max-width:100%;margin-top:16px}
button{margin-top:16px;padding:8px 18px}#s{margin-left:10px;color:#666}</style></head><body>
<h2>Head swap - Flux 2 Klein 4B + BFS</h2>
<form id="f"><label>API token</label><input id="t" type="password" size="40" required>
<label>Picture 1 - body / target image</label><input name="body" type="file" accept="image/*" required>
<label>Picture 2 - head / face to insert</label><input name="head" type="file" accept="image/*" required>
<label>LoRA strength</label><input name="lora_strength" type="number" step="0.05" min="0" max="1.5" value="1.0">
<label>LoRA version</label><select name="lora"><option>v1</option><option>v1.1</option></select>
<label>Mode</label><select name="mode"><option value="crop">crop - swap head, keep full-res original</option><option value="full">full - regenerate whole image</option></select>
<label>Seed</label><input name="seed" type="number" value="42">
<label>Steps</label><input name="steps" type="number" min="1" max="12" value="4">
<label>Extra prompt (optional)</label><input name="extra_prompt" size="60" placeholder="e.g. the face remains unchanged, the eyes remain unchanged">
<br><button>Swap</button><span id="s"></span></form><img id="o">
<script>
const f=document.getElementById('f'),s=document.getElementById('s'),o=document.getElementById('o');
f.onsubmit=async e=>{e.preventDefault();s.textContent='Working... (first run after idle takes ~1-2 min)';o.removeAttribute('src');
const r=await fetch('headswap',{method:'POST',headers:{Authorization:'Bearer '+document.getElementById('t').value},body:new FormData(f)});
if(!r.ok){s.textContent='Error '+r.status+': '+await r.text();return}
o.src=URL.createObjectURL(await r.blob());s.textContent='Done';};
</script></body></html>"""


@app.function(image=image, secrets=[modal.Secret.from_name("klein-api-token")], timeout=900)
@modal.asgi_app()
def api():
    import hmac

    from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
    from fastapi.responses import HTMLResponse, Response

    web = FastAPI(title="Klein Head Swap")
    token = os.environ["API_TOKEN"]

    @web.get("/", response_class=HTMLResponse)
    def index():
        return PAGE

    @web.post("/headswap")
    async def headswap(body: UploadFile = File(...), head: UploadFile = File(...), lora_strength: float = Form(1.0),
                       steps: int = Form(4), seed: int = Form(42), extra_prompt: str = Form(""),
                       max_side: int = Form(1024), lora: str = Form("v1"), mode: str = Form("crop"), authorization: str = Header("")):
        if not hmac.compare_digest(authorization, f"Bearer {token}"):
            raise HTTPException(401, "bad or missing token")
        if not (0 <= lora_strength <= 2 and 1 <= steps <= 50 and 256 <= max_side <= 2048 and lora in LORAS and mode in ("crop", "full")):
            raise HTTPException(400, "parameter out of range")
        png = await KleinHeadSwap().swap.remote.aio(await body.read(), await head.read(), lora_strength, steps,
                                                    seed, extra_prompt, max_side, lora, mode)
        return Response(png, media_type="image/png")

    return web


@app.local_entrypoint()
def main(body: str, head: str, out: str = "klein_headswap.png", lora_strength: float = 1.0, steps: int = 4,
         seed: int = 42, extra_prompt: str = "", lora: str = "v1", mode: str = "crop"):
    png = KleinHeadSwap().swap.remote(open(body, "rb").read(), open(head, "rb").read(), lora_strength, steps, seed,
                                      extra_prompt, 1024, lora, mode)
    open(out, "wb").write(png)
    print("saved", out)
