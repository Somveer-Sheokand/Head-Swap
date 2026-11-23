"""
Qwen-Image-2.1 on Modal: an open-weight, ChatGPT-image-style model
(text-to-image + multi-reference image editing + RGBA) served on GPUs.

  modal run qwen_image_modal.py::download      # one-time: cache weights in a Volume (CPU only)
  modal deploy qwen_image_modal.py             # deploy GPU class + OpenAI-compatible HTTP API
  modal run benchmark.py                       # performance test against the deployment

Variants:
  turbo -> Qwen/Qwen-Image-2.1-Turbo  (8 steps, distilled, CFG=1)
  base  -> Qwen/Qwen-Image-2.1        (40 steps)
"""

import io
import os
import time

import modal

APP_NAME = "qwen-image-2-1"
MODELS = {
    "turbo": "Qwen/Qwen-Image-2.1-Turbo",
    "base": "Qwen/Qwen-Image-2.1",
}
DEFAULT_STEPS = {"turbo": 8, "base": 40}
DEFAULT_GPU = os.environ.get("QWEN_GPU", "H100")
# Pinned diffusers main commit containing QwenImage21Pipeline + sample_sigmas support (PR #14950).
DIFFUSERS_COMMIT = "750b3be2c0198e194c87e931a6904066de167a5f"
CACHE_DIR = "/cache"

app = modal.App(APP_NAME)
hf_cache = modal.Volume.from_name("qwen-image-hf-cache", create_if_missing=True)
uploads = modal.Dict.from_name("qwen-headswap-uploads", create_if_missing=True)  # images served at /files/{key}

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .uv_pip_install(
        "torch==2.14.1",
        "torchvision==0.29.2",
        "transformers==5.24.0",
        "accelerate==1.16.0",
        "huggingface_hub==2.4.0",
        "safetensors",
        "pillow",
        "numpy",
        "fastapi[standard]",
        f"diffusers @ git+https://github.com/huggingface/diffusers@{DIFFUSERS_COMMIT}",
    )
    .env({"HF_HOME": CACHE_DIR, "HF_XET_HIGH_PERFORMANCE": "1"})
)


# --------------------------------------------------------------------------- weights


@app.function(image=image, volumes={CACHE_DIR: hf_cache}, timeout=3 * 3600, cpu=8, memory=16384)
def download(variant: str = "all"):
    """Pull weights into the shared Volume so GPU containers never download."""
    from huggingface_hub import snapshot_download

    keys = list(MODELS) if variant == "all" else [variant]
    for key in keys:
        t0 = time.perf_counter()
        path = snapshot_download(MODELS[key], ignore_patterns=["assets/*", "*.png", "*.jpg", "*.gif"])
        print(f"{MODELS[key]} -> {path} in {time.perf_counter() - t0:.0f}s")
    hf_cache.commit()


# --------------------------------------------------------------------------- GPU model


def _png(img) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def preserve_original(target_bytes: bytes, generated_png: bytes) -> bytes:
    """Return the target at its ORIGINAL size where only the region the model actually changed (the head)
    comes from the generated image. Everything else (clothes, body, background, text) stays pixel-identical.

    1. resize the generated image to the target size and undo the model's small global drift
    2. find where it differs clearly from the target, keep the largest changed blob (the head), fill it
    3. feather that mask and paste the generated pixels only there
    """
    import numpy as np
    from PIL import Image, ImageFilter
    from scipy import ndimage

    tpl_src = Image.open(io.BytesIO(target_bytes))
    tpl = tpl_src.convert("RGB")
    gen = Image.open(io.BytesIO(generated_png)).convert("RGB").resize(tpl.size, Image.LANCZOS)

    # work on a ~1024px copy for speed
    s = 1024 / max(tpl.size)
    small = (max(1, round(tpl.width * s)), max(1, round(tpl.height * s)))
    a = np.asarray(tpl.resize(small, Image.BILINEAR).filter(ImageFilter.GaussianBlur(1.5)), np.float32)
    b_img = gen.resize(small, Image.BILINEAR).filter(ImageFilter.GaussianBlur(1.5))

    # global drift: best integer shift (in small pixels) within +-4
    best, best_err = (0, 0), None
    ga = a.mean(2)
    gb = np.asarray(b_img, np.float32).mean(2)
    for dy in range(-4, 5):
        for dx in range(-4, 5):
            err = np.abs(ga[8:-8, 8:-8] - np.roll(gb, (-dy, -dx), (0, 1))[8:-8, 8:-8]).mean()
            if best_err is None or err < best_err:
                best, best_err = (dx, dy), err
    dx, dy = best
    if dx or dy:
        gen = gen.transform(gen.size, Image.AFFINE, (1, 0, dx / s, 0, 1, dy / s), resample=Image.BICUBIC)
        b_img = gen.resize(small, Image.BILINEAR).filter(ImageFilter.GaussianBlur(1.5))
    b = np.asarray(b_img, np.float32)

    d = np.abs(a - b).mean(2)
    changed = ndimage.gaussian_filter(np.clip((d - 22) / 16, 0, 1), 5) > 0.2
    # 1) head core: cut thin strips (frame edges, clothing outlines) the model re-drew next to the head so they
    #    don't join the head blob; the head itself is a big compact area and survives the opening
    core = ndimage.binary_opening(changed, structure=np.ones((3, 3)), iterations=12)
    labels, n = ndimage.label(core)
    if n == 0:
        return _png(tpl_src)
    sizes = ndimage.sum(core, labels, range(1, n + 1))
    head = labels == (int(np.argmax(sizes)) + 1)
    # 2) border: add back every (even faintly) changed pixel right next to the core, so the old head's outline
    #    (ears, hair edge) is fully replaced and no ghost of it remains; far-away strips stay original
    faint = ndimage.gaussian_filter(np.clip((d - 10) / 16, 0, 1), 2) > 0.2
    head = head | (faint & ndimage.binary_dilation(head, iterations=14))
    head = ndimage.binary_fill_holes(ndimage.binary_closing(head, iterations=6))
    head = ndimage.binary_dilation(head, iterations=3)

    mask = Image.fromarray((head * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(4))
    mask = mask.resize(tpl.size, Image.BILINEAR).filter(ImageFilter.GaussianBlur(6))
    out = tpl.copy()
    out.paste(gen, (0, 0), mask)
    if tpl_src.mode == "RGBA":
        out.putalpha(tpl_src.getchannel("A"))
    return _png(out)


@app.cls(
    image=image,
    gpu=DEFAULT_GPU,
    volumes={CACHE_DIR: hf_cache},
    timeout=1800,
    # Idle GPU time is billed: 60s keeps back-to-back requests warm without paying 5 min per burst.
    scaledown_window=int(os.environ.get("QWEN_SCALEDOWN_S", "60")),
    max_containers=4,  # spend cap for load tests
)
class QwenImage:
    variant: str = modal.parameter(default="turbo")

    @modal.enter()
    def load(self):
        import torch
        from diffusers import QwenImage21Pipeline

        t0 = time.perf_counter()
        self.pipe = QwenImage21Pipeline.from_pretrained(MODELS[self.variant], torch_dtype=torch.bfloat16).to("cuda")
        torch.cuda.synchronize()
        self.load_s = time.perf_counter() - t0
        self.weights_gb = torch.cuda.memory_allocated() / 1e9
        self.gpu_name = torch.cuda.get_device_name(0)
        self.vram_total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        self.served = 0
        print(f"[{self.variant}] loaded on {self.gpu_name} in {self.load_s:.1f}s, {self.weights_gb:.1f} GB resident")

    @modal.method()
    def info(self) -> dict:
        return {
            "variant": self.variant,
            "model": MODELS[self.variant],
            "gpu": self.gpu_name,
            "vram_total_gb": round(self.vram_total_gb, 1),
            "weights_gb": round(self.weights_gb, 2),
            "load_s": round(self.load_s, 2),
            "served": self.served,
        }

    @modal.method()
    def generate(
        self,
        prompt: str,
        width: int = 1024,
        height: int = 1024,
        steps: int | None = None,
        seed: int = 0,
        images: list[bytes] | None = None,
        use_kv_cache: bool = True,
        negative_prompt: str | None = None,
        true_cfg_scale: float = 1.0,
    ) -> dict:
        import torch
        from PIL import Image

        cond = [Image.open(io.BytesIO(b)) for b in images] if images else None
        step_ends: list[float] = []

        def on_step(pipe, i, t, kw):
            torch.cuda.synchronize()
            step_ends.append(time.perf_counter())
            return kw

        kwargs = dict(
            prompt=prompt,
            image=cond,
            width=width,
            height=height,
            generator=torch.Generator("cpu").manual_seed(seed),
            use_kv_cache=use_kv_cache,
            callback_on_step_end=on_step,
            true_cfg_scale=true_cfg_scale,
            negative_prompt=negative_prompt,
            output_resolution=max(width, height),
        )
        # Turbo carries its own 8-step sigma schedule; num_inference_steps only matters for base.
        if self.variant == "base":
            kwargs["num_inference_steps"] = steps or DEFAULT_STEPS["base"]

        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        try:
            out = self.pipe(**kwargs).images[0]
        except Exception as e:  # re-raise as a plain error so clients without torch can read it
            torch.cuda.empty_cache()
            raise RuntimeError(f"{type(e).__name__}: {str(e)[:600]}") from None
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        png = _png(out)
        t2 = time.perf_counter()
        self.served += 1

        n = len(step_ends)
        later = [b - a for a, b in zip(step_ends, step_ends[1:])]
        metrics = {
            "variant": self.variant,
            "gpu": self.gpu_name,
            "width": out.width,
            "height": out.height,
            "mode": out.mode,
            "steps": n,
            "n_ref_images": len(cond) if cond else 0,
            "pipeline_s": round(t1 - t0, 3),
            # encode prompt (+ ref images) + latents prep + first (prefill) step
            "first_step_s": round(step_ends[0] - t0, 3) if n else None,
            "step_mean_s": round(sum(later) / len(later), 4) if later else None,
            "decode_s": round(t1 - step_ends[-1], 3) if n else None,  # VAE decode + postprocess
            "png_encode_s": round(t2 - t1, 3),
            "png_mb": round(len(png) / 1e6, 2),
            "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
            "container_request_no": self.served,
            "container_id": os.environ.get("MODAL_TASK_ID", "?"),
        }
        return {"png": png, "metrics": metrics}


# --------------------------------------------------------------------------- HTTP API
# OpenAI-Images-compatible surface so existing gpt-image clients can point at it:
#   POST /v1/images/generations   JSON  {prompt, size:"1024x1024", n, model, seed}
#   POST /v1/images/edits         multipart  image[] files + prompt (+ size, model)


#   POST /v1/headswap             multipart  target_image + source_image (+ prompt)  -> {job_id}
#   GET  /v1/headswap/{job_id}    -> {status: pending|completed|failed, download_url, metrics}
#   GET  /v1/headswap/{job_id}/image   final PNG once completed (the download_url)
#   GET  /v1/headswap/prompt      default head-swap prompt (override per request)
# With preserve_original=true (default) the result is returned at the target's ORIGINAL size and only the
# head region the model changed is taken from the generation; clothes/body/background stay pixel-identical.
# No authorization: the API is public (GPU spend capped by max_containers).

MIN_SIDE, MAX_SIDE = 256, 2048
MAX_UPLOAD_BYTES = 20 * 1024 * 1024

# Image 1 = target (body/scene kept), image 2 = source (face taken).
DEFAULT_HEADSWAP_PROMPT = (
    "Keep the central model's head in the first image exactly where and how it is, with the same "
    'orientation, tilt, chin height and eye direction, and change only the facial identity on it: the '
    'eyes, eyebrows, nose, lips, face shape, skin tone and age become those of the person in the second '
    'image, so the person is instantly recognisable.\n\n'
    'Copy the facial hair exactly as it is in the second image, with the same amount, length and shape, '
    'and add none that is not there. Realistic, unretouched skin texture with visible pores, not '
    'airbrushed, not beautified and not made younger. Keep the original facial expression and mouth '
    'shape, and the lip colour and makeup style of the first image.\n\n'
    'The lighting on the face stays exactly as in the first image: the same light direction, highlights, '
    'shadows, contrast and colour grading. The face, neck and chest are one continuous skin tone.\n\n'
    'Preserve the original hairstyle, head size, head position, both ears and everything worn on them, '
    'the neck and anything worn around it, body pose, clothing, background, typography and branding '
    'exactly as in the first image. No changes outside the face.\n\n'
    'Seamless photorealistic integration, premium fashion editorial photograph.'
)

cpu_image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy", "pillow", "scipy")


@app.function(image=cpu_image, cpu=2, memory=4096, timeout=1800)
def headswap_job(target: bytes, source: bytes, variant: str, gen: dict, preserve: bool = True) -> dict:
    """One head-swap job: GPU generation, then (optionally) restore the original outside the head."""
    from PIL import Image

    tgt = Image.open(io.BytesIO(target))
    gen_target = target if tgt.mode == "RGB" else _png(tgt.convert("RGB"))
    res = QwenImage(variant=variant).generate.remote(images=[gen_target, source], **gen)
    t0 = time.perf_counter()
    png = preserve_original(target, res["png"]) if preserve else res["png"]
    metrics = dict(res["metrics"], preserve_original=preserve, compose_s=round(time.perf_counter() - t0, 2))
    metrics["output_width"], metrics["output_height"] = Image.open(io.BytesIO(png)).size
    return {"png": png, "metrics": metrics}


def _variant_for(model: str | None) -> str:
    m = (model or "").lower()
    return "base" if m in ("base", "qwen-image-2.1", MODELS["base"].lower()) else "turbo"


def _round16(x: float) -> int:
    return max(MIN_SIDE, min(MAX_SIDE, int(round(x / 16)) * 16))


def _size(size: str | None) -> tuple[int, int]:
    """Parse 'WxH'; raises ValueError on bad input or out-of-range sizes."""
    if not size or size == "auto":
        return 1024, 1024
    try:
        w, h = (int(v) for v in size.lower().split("x"))
    except ValueError:
        raise ValueError(f"size must look like '1024x1024', got {size!r}") from None
    if not (MIN_SIDE <= w <= MAX_SIDE and MIN_SIDE <= h <= MAX_SIDE):
        raise ValueError(f"width and height must be between {MIN_SIDE} and {MAX_SIDE}")
    return _round16(w), _round16(h)


def _auto_size_from(img_bytes: bytes, long_side: int = 1024) -> tuple[int, int]:
    """Keep the target photo's aspect ratio, longest side = long_side, multiples of 16."""
    from PIL import Image

    w, h = Image.open(io.BytesIO(img_bytes)).size
    s = long_side / max(w, h)
    return _round16(w * s), _round16(h * s)


@app.function(
    image=image,
    scaledown_window=300,
    timeout=1800,
)
@modal.concurrent(max_inputs=32)
@modal.asgi_app()
def api():
    import asyncio
    import base64

    from fastapi import FastAPI, File, Form, HTTPException, Request, Response, UploadFile
    from pydantic import BaseModel

    web = FastAPI(title="Qwen-Image-2.1 on Modal", version="1.2")
    auth = []  # public API: no authorization required

    async def _read_upload(f: UploadFile, name: str) -> bytes:
        data = await f.read()
        if not data:
            raise HTTPException(400, f"{name} is empty")
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, f"{name} is larger than {MAX_UPLOAD_BYTES // 2**20} MB")
        try:
            from PIL import Image

            Image.open(io.BytesIO(data)).verify()
        except Exception:
            raise HTTPException(400, f"{name} is not a valid image") from None
        return data

    async def _image_input(upload: UploadFile | None, url: str | None, name: str) -> bytes:
        """Accept either an uploaded file or an http(s) URL to download."""
        if upload is not None and upload.filename:
            return await _read_upload(upload, name)
        if not url:
            raise HTTPException(400, f"send {name} as a file or {name}_url as an https link")
        if not url.lower().startswith(("https://", "http://")):
            raise HTTPException(400, f"{name}_url must start with https://")
        import httpx

        try:
            async with httpx.AsyncClient(follow_redirects=True, timeout=60) as client:
                r = await client.get(url)
                r.raise_for_status()
        except Exception as e:
            raise HTTPException(400, f"could not download {name}_url: {e}") from None
        data = r.content
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, f"{name}_url is larger than {MAX_UPLOAD_BYTES // 2**20} MB")
        try:
            from PIL import Image

            Image.open(io.BytesIO(data)).verify()
        except Exception:
            raise HTTPException(400, f"{name}_url is not a valid image") from None
        return data

    def _parse_size(size):
        try:
            return _size(size)
        except ValueError as e:
            raise HTTPException(400, str(e)) from None

    class GenReq(BaseModel):
        prompt: str
        model: str | None = "qwen-image-2.1-turbo"
        size: str | None = "1024x1024"
        n: int = 1
        seed: int | None = None
        response_format: str | None = "b64_json"

    async def _run(variant, prompt, w, h, n, seed, images=None):
        if n < 1 or n > 4:
            raise HTTPException(400, "n must be 1..4")
        cls = QwenImage(variant=variant)
        base_seed = seed if seed is not None else int(time.time()) % 2**31
        t0 = time.perf_counter()
        try:
            results = await asyncio.gather(
                *[
                    cls.generate.remote.aio(prompt=prompt, width=w, height=h, seed=base_seed + i, images=images)
                    for i in range(n)
                ]
            )
        except Exception as e:
            raise HTTPException(500, f"generation failed: {e}") from None
        return {
            "created": int(time.time()),
            "data": [{"b64_json": base64.b64encode(r["png"]).decode(), "revised_prompt": None} for r in results],
            "usage": {"server_wall_s": round(time.perf_counter() - t0, 3)},
            "metrics": [r["metrics"] for r in results],
        }

    @web.get("/health")
    async def health():
        return {"ok": True, "models": MODELS}

    # ------------------------------------------------------------------ image hosting (for *_url inputs)

    @web.post("/upload")
    async def upload(request: Request, file: UploadFile = File(...)):
        """Store an image and return a public https URL usable as image_url / source_face_url."""
        import uuid

        data = await _read_upload(file, "file")
        ext = os.path.splitext(file.filename or "")[1].lower() or ".png"
        if ext not in (".png", ".jpg", ".jpeg", ".webp"):
            ext = ".png"
        key = uuid.uuid4().hex + ext
        await uploads.put.aio(key, data)
        return {"url": f"https://{request.headers['host']}/files/{key}"}

    @web.get("/files/{key}")
    async def get_file(key: str):
        data = await uploads.get.aio(key)
        if data is None:
            raise HTTPException(404, "file not found")
        media = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}.get(
            os.path.splitext(key)[1], "image/png"
        )
        return Response(content=data, media_type=media)

    @web.post("/v1/images/generations", dependencies=auth)
    async def generations(req: GenReq):
        w, h = _parse_size(req.size)
        return await _run(_variant_for(req.model), req.prompt, w, h, req.n, req.seed)

    @web.post("/v1/images/edits", dependencies=auth)
    async def edits(
        prompt: str = Form(...),
        image: list[UploadFile] = File(...),
        model: str | None = Form("qwen-image-2.1-turbo"),
        size: str | None = Form("1024x1024"),
        n: int = Form(1),
        seed: int | None = Form(None),
    ):
        if len(image) > 10:
            raise HTTPException(400, "up to 10 reference images")
        refs = [await _read_upload(f, f"image[{i}]") for i, f in enumerate(image)]
        w, h = _parse_size(size)
        return await _run(_variant_for(model), prompt, w, h, n, seed, images=refs)

    # ------------------------------------------------------------------ head swap (async jobs)

    @web.get("/v1/headswap/prompt", dependencies=auth)
    async def headswap_prompt():
        return {
            "default_prompt": DEFAULT_HEADSWAP_PROMPT,
            "image_order": {"first": "target_image (body/scene kept)", "second": "source_image (head taken)"},
            "how_to_customize": "Send `prompt` to replace the default entirely, or `extra_prompt` to append "
            "instructions to it (e.g. 'keep the original beard', 'make the hair shorter').",
        }

    @web.post("/v1/headswap", dependencies=auth, status_code=202)
    async def headswap_submit(
        request: Request,
        target_image: UploadFile | None = File(None, description="Photo whose body/pose/background are kept"),
        source_image: UploadFile | None = File(None, description="Photo whose head/face is swapped in"),
        target_image_url: str | None = Form(None, description="https URL instead of uploading target_image"),
        source_image_url: str | None = Form(None, description="https URL instead of uploading source_image"),
        prompt: str | None = Form(None, description="Full prompt; replaces the default"),
        extra_prompt: str | None = Form(None, description="Extra instructions appended to the prompt"),
        size: str | None = Form("auto", description="Generation size: 'auto' (target aspect, up to 2048) or 'WxH'"),
        seed: int | None = Form(None),
        model: str | None = Form("turbo", description="'turbo' (fast) or 'base' (higher quality, slower)"),
        steps: int | None = Form(None, description="Base model only; default 40"),
        guidance: float | None = Form(None, description="true_cfg_scale; default 1.0 turbo / 4.0 base"),
        negative_prompt: str | None = Form(None),
        preserve_original: bool = Form(
            True, description="Return at the target's original size, changing only the head region"
        ),
    ):
        target = await _image_input(target_image, target_image_url, "target_image")
        source = await _image_input(source_image, source_image_url, "source_image")
        if not size or size == "auto":
            from PIL import Image

            cap = 1600 if _variant_for(model) == "base" else MAX_SIDE  # base + 2 refs OOMs above 1600 on H100
            long_side = min(cap, max(Image.open(io.BytesIO(target)).size))
            w, h = _auto_size_from(target, long_side)
        else:
            w, h = _parse_size(size)
        variant = _variant_for(model)
        if steps is not None and not (1 <= steps <= 100):
            raise HTTPException(400, "steps must be 1..100")

        final_prompt = (prompt or DEFAULT_HEADSWAP_PROMPT).strip()
        if extra_prompt and extra_prompt.strip():
            final_prompt = f"{final_prompt} {extra_prompt.strip()}"
        if guidance is None:
            guidance = 4.0 if variant == "base" else 1.0
        if guidance > 1.0 and not negative_prompt:
            negative_prompt = " "  # CFG needs an unconditional branch

        gen = dict(
            prompt=final_prompt,
            width=w,
            height=h,
            steps=steps,
            seed=seed if seed is not None else int(time.time()) % 2**31,
            negative_prompt=negative_prompt,
            true_cfg_scale=guidance,
        )
        call = await headswap_job.spawn.aio(target, source, variant, gen, preserve_original)
        return {
            "job_id": call.object_id,
            "status": "pending",
            "model": variant,
            "generation_size": f"{w}x{h}",
            "preserve_original": preserve_original,
            "prompt_used": final_prompt,
            "status_url": f"https://{request.headers['host']}/v1/headswap/{call.object_id}",
        }

    class HeadswapJSON(BaseModel):
        image_url: str | None = None  # template / poster (alias of target_image_url)
        target_image_url: str | None = None
        source_face_url: str
        prompt: str | None = None
        extra_prompt: str | None = None
        seed: int | None = None
        model: str | None = "base"  # best quality; "turbo" is faster
        size: str | None = "auto"
        preserve_original: bool = True

    @web.post("/jobs", dependencies=auth, status_code=202)
    @web.post("/headswap", dependencies=auth, status_code=202)
    async def headswap_json(req: HeadswapJSON, request: Request):
        """JSON version: {"image_url", "source_face_url", "prompt"} -> {job_id, status_url}."""
        target_url = req.image_url or req.target_image_url
        if not target_url:
            raise HTTPException(400, "image_url is required")
        res = await headswap_submit(
            request=request,
            target_image=None,
            source_image=None,
            target_image_url=target_url,
            source_image_url=req.source_face_url,
            prompt=req.prompt,
            extra_prompt=req.extra_prompt,
            size=req.size,
            seed=req.seed,
            model=req.model,
            steps=None,
            guidance=None,
            negative_prompt=None,
            preserve_original=req.preserve_original,
        )
        return {
            "job_id": res["job_id"],
            "status": "pending",
            "status_url": f"https://{request.headers['host']}/jobs/{res['job_id']}",
        }

    @web.get("/jobs/{job_id}", dependencies=auth)
    @web.get("/status/{job_id}", dependencies=auth)
    async def status_short(job_id: str, request: Request):
        full = await headswap_status(job_id, request)
        return {k: full[k] for k in ("job_id", "status", "download_url", "error") if k in full}

    async def _job_result(job_id: str):
        """Returns (status, result_or_error)."""
        try:
            call = modal.FunctionCall.from_id(job_id)
            return "completed", await call.get.aio(timeout=0)
        except (TimeoutError, modal.exception.TimeoutError):
            return "pending", None
        except modal.exception.OutputExpiredError:
            raise HTTPException(410, "job result expired") from None
        except (modal.exception.NotFoundError, modal.exception.InvalidError):
            raise HTTPException(404, "unknown job_id") from None
        except Exception as e:
            return "failed", str(e)

    @web.get("/v1/headswap/{job_id}", dependencies=auth)
    async def headswap_status(job_id: str, request: Request):
        status, res = await _job_result(job_id)
        body = {"job_id": job_id, "status": status}
        if status == "completed":
            body["download_url"] = f"https://{request.headers['host']}/v1/headswap/{job_id}/image"
            body["metrics"] = res["metrics"]
        elif status == "failed":
            body["error"] = res
        return body

    @web.get("/v1/headswap/{job_id}/image", dependencies=auth)
    async def headswap_image(job_id: str):
        status, res = await _job_result(job_id)
        if status != "completed":
            raise HTTPException(409 if status == "pending" else 500, f"job is {status}" + (f": {res}" if res else ""))
        return Response(content=res["png"], media_type="image/png")

    return web


@app.local_entrypoint()
def smoke(prompt: str = 'A neon shop sign that reads "MODAL x QWEN", rainy night', variant: str = "turbo"):
    """Quick end-to-end check: modal run qwen_image_modal.py --prompt '...'"""
    r = QwenImage(variant=variant).generate.remote(prompt=prompt)
    with open(f"smoke_{variant}.png", "wb") as f:
        f.write(r["png"])
    print(r["metrics"])
