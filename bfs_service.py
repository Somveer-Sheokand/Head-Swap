"""BFS head-swap service on Modal.

Model: Qwen Image 2.1 + BFS LoRA bfs_head_v1.1_qwen_2.1 (Alissonerdx/BFS-Best-Face-Swap) via ComfyUI.
Pipeline: detect target head per frame (InsightFace) -> crop head -> BFS on crop -> feathered paste-back
          + colour match -> re-encode with original audio.
"""
import json, os, subprocess, time, urllib.request, urllib.parse, uuid, io
import modal

app = modal.App("bfs-headswap-api")
models_vol = modal.Volume.from_name("bfs-models", create_if_missing=True)
jobs_vol = modal.Volume.from_name("bfs-jobs", create_if_missing=True)
MODELS, JOBS = "/models", "/jobs"
GPU_USD_PER_SEC = 1.95 / 3600  # L40S, approx
# quality presets used by every endpoint ("best" = previous behaviour)
PRESETS = {
    "best": dict(res=1024, href=768, key_step=1, taps=2),
    # adaptive: small / far-away heads are generated at a smaller size (never below MIN_RES) - they are shrunk
    # to their on-screen size anyway, so the extra pixels cost GPU time without being visible
    "fast": dict(res=832, href=640, key_step=2, taps=2, adaptive=True),
}
MIN_RES = 640
EST_FACTOR = 2.0                     # real Modal bills were ~2x the plain per-frame estimate (cold starts, idle GPU time)

# Speed-ups that are built in but OFF until they have been tested once on a real job (each needs a redeploy):
USE_SNAPSHOT = False   # Modal memory + GPU snapshot of the loaded ComfyUI -> cold start ~60-110 s -> ~10-20 s
USE_SAGE = False       # SageAttention in ComfyUI -> ~15-30 % less GPU time per frame
BLEND_CHUNK, BLEND_PAD = 24, 6       # frames per parallel blend container, overlap frames each side
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

cpu_image = cpu_image.run_commands(
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


def _graph(body, head, seed, steps, sampler, res=BODY_RES, href=HEAD_RES, prev=None):
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
    if prev:
        g["16"] = {"class_type": "LoadImage", "inputs": {"image": prev}}
        g["17"] = {"class_type": "ImageScale", "inputs": {"image": ["16", 0], "upscale_method": "lanczos", "width": res, "height": res, "crop": "disabled"}}
        g["11"]["inputs"]["images.image_3"] = ["17", 0]
        g["11"]["inputs"]["prompt"] = PROMPT3
    return g


_SNAP_OPTS = dict(enable_memory_snapshot=True, experimental_options={"enable_gpu_snapshot": True}) if USE_SNAPSHOT else {}


# no min_containers: the GPU scales to zero 15 s after the last job, so an idle app costs nothing
@app.cls(image=gpu_image, gpu="L40S", volumes={MODELS: models_vol}, timeout=1800, scaledown_window=15, max_containers=4, **_SNAP_OPTS)
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
    def process(self, batch: list, head_png: bytes, steps: int = 8, sampler: str = "euler", seed: int = 42, res: int = BODY_RES, href: int = HEAD_RES):
        """batch: [(idx, crop_png_bytes)] -> [(idx, out_png_bytes)]"""
        os.makedirs("/comfy/input", exist_ok=True)
        hname = f"head_{uuid.uuid4().hex[:8]}.png"
        open(f"/comfy/input/{hname}", "wb").write(head_png)
        results = []
        t0 = time.time()
        for idx, b in batch:
            bname = f"b_{uuid.uuid4().hex[:8]}.png"
            open(f"/comfy/input/{bname}", "wb").write(b)
            pid = self._post({"prompt": _graph(bname, hname, seed, steps, sampler, res, href)})["prompt_id"]
            while True:
                h = json.load(urllib.request.urlopen(f"http://127.0.0.1:8188/history/{pid}"))
                if pid in h:
                    break
                time.sleep(0.5)
            if h[pid]["status"].get("status_str") != "success":
                raise RuntimeError("comfy failed: " + json.dumps(h[pid]["status"])[:800])
            im = h[pid]["outputs"]["15"]["images"][0]
            q = urllib.parse.urlencode({"filename": im["filename"], "subfolder": im["subfolder"], "type": im["type"]})
            results.append((idx, urllib.request.urlopen(f"http://127.0.0.1:8188/view?{q}").read()))
            os.remove(f"/comfy/input/{bname}")
        print(f"batch of {len(batch)} done in {time.time()-t0:.1f}s", flush=True)
        return results


# ----------------------------------------------------------------------------- CPU helpers
def _fa():
    from insightface.app import FaceAnalysis
    fa = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"], allowed_modules=["detection", "recognition"])
    fa.prepare(ctx_id=-1, det_size=(640, 640))
    return fa


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


def _smooth_track(track, n, max_gap=6):
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
        k = np.exp(-0.5 * (np.arange(-4, 5) / 2.0) ** 2); k /= k.sum()
        pad = np.pad(arr, ((4, 4), (0, 0)), mode="edge")
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


@app.function(image=cpu_image, volumes={JOBS: jobs_vol}, cpu=1, memory=2048, timeout=5400)
def swap_job(template_id: str, source_face_url: str, quality: str = "best", start_frame: int = 0, max_frames: int = 0,
             max_cost_usd: float = 5.0, seed: int = 42, steps: int = 8, sampler: str = "euler", output_put_url: str = "",
             dry_run: bool = False, keep_raw: bool = False):
    """Video head swap: GPU stage once, then blending split over many small CPU containers, then join + audio."""
    import shutil
    t0 = time.time()
    preset = dict(PRESETS.get(quality, PRESETS["best"]))
    if preset.pop("adaptive", False):
        preset["res"] = _adaptive_res(template_id, start_frame, max_frames, preset["res"])
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
                    "-map", "0:v", "-map", "1:a?", "-c:v", "copy", "-c:a", "aac", "-b:a", "160k", "-shortest", out_path], check=True)
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
@app.function(image=cpu_image, volumes={JOBS: jobs_vol}, secrets=[modal.Secret.from_name("bfs-api-token")], timeout=600)
@modal.asgi_app(label="bfs-headswap-api")
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
        kw = {k: body[k] for k in ("target_face_url", "face_index", "quality", "seed", "output_put_url", "max_cost_usd") if k in body}
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


def _head_region(sess, bgr, fx, fy, fw, fh, cap):
    """soft mask (0..1, same size as bgr) of the head+hair that contains the target face centre"""
    import cv2, numpy as np
    res = bgr.shape[0]
    lab = _parse_labels(sess, bgr)
    head = np.isin(lab, HEAD_CLASSES).astype(np.uint8)
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


@app.function(image=cpu_image, volumes={JOBS: jobs_vol}, cpu=4, memory=8192, timeout=1800, max_containers=12)
def blend_part(*a, **kw):
    """Blends one slice of the video from already generated heads (reuse_raw) - many run in parallel."""
    return _swap2_impl(*a, **kw)


def _swap2_impl(template_id: str, source_face_url: str, start_frame: int = 0, max_frames: int = 0, res: int = 896, href: int = 704,
          max_cost_usd: float = 1.2, seed: int = 42, steps: int = 8, sampler: str = "euler", dry_run: bool = False,
          reuse_raw: str = "", promote_thr: float = 6.0, key_step: int = 2, grain: bool = False, grain_max: float = 2.5, fix_top: bool = True, align_fix: bool = False, align_sigma: float = 4.0, taps: int = 1, seg_mask: bool = True, seg_full: bool = False, output_put_url: str = "", flow_comp: bool = False, soft_match: bool = False, seq_ref: int = 0,
                gpu_only: bool = False, core: list = None, out_name: str = ""):
    import cv2, numpy as np
    t0 = time.time()
    jobs_vol.reload()
    d = f"{JOBS}/{template_id}"
    meta = json.load(open(f"{d}/track.json"))
    n, fps, W, H = meta["frames"], meta["fps"], meta["w"], meta["h"]
    end = n if not max_frames else min(n, start_frame + max_frames)
    sm = _smooth_track(meta["track"], n)
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
    for r in runs:
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
        keys = sorted(k for k in (int(x[:4]) for x in os.listdir(f"{d}/{reuse_raw}") if x.endswith(".png")) if k in frames and sm[k] is not None)
    keyset = set(keys)
    per = 14.0 * (res / 896) ** 2
    est = (len(keys) * per + max(1, min(4, round(len(keys) / 40))) * 110) * GPU_USD_PER_SEC * EST_FACTOR
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
        n_gpu = max(1, min(4, round(len(crops) / 40)))
        size = max(1, min(100, -(-len(crops) // n_gpu)))
        batches = [crops[k:k + size] for k in range(0, len(crops), size)]
        for rs in Worker().process.map(batches, kwargs={"head_png": head_png, "steps": steps, "sampler": sampler, "seed": seed, "res": res, "href": href}):
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
        sig_s = {}
        for r in runs:
            kr_ = [k for k in r if k in SIG]
            for q_, k in enumerate(kr_):
                win_ = [SIG[x] for x in kr_[max(0, q_ - 4):q_ + 5]]
                sig_s[k] = float(np.median(win_)) if win_ else 0.0
        ksort = sorted(keys)
        for i in keys:
            nb = [k for k in (i - 1, i + 1) if k in MW and k in keyset]
            m = MW[i] * 0.5 + sum(MW[k] * (0.5 / len(nb)) for k in nb) if nb else MW[i]
            m = np.clip(m, 0, 1).astype(np.float32)
            CM[i] = m
            gwi = cv2.GaussianBlur(GW[i], (0, 0), sig_s[i]) if (soft_match and sig_s.get(i, 0) > 0.2) else GW[i]
            Kc[i] = (frames[i].astype(np.float32) * (1 - m[..., None]) + gwi.astype(np.float32) * m[..., None]).clip(0, 255).astype(np.uint8)
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
def measure(template_id: str, names: list):
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
        pos, i = {}, 0
        while True:
            ok, fr = cap.read()
            if not ok:
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
               seed: int = 42, steps: int = 8, sampler: str = "euler", output_put_url: str = "", max_cost_usd: float = 0.5):
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
    out = Worker().process.remote([(0, cv2.imencode(".png", crop)[1].tobytes())], head_png, steps=steps, sampler=sampler, seed=seed, res=res, href=href)
    gen = cv2.imdecode(np.frombuffer(out[0][1], np.uint8), cv2.IMREAD_COLOR)

    sess = _seg_session(False)
    fx, fy = res / 2 + sc_ * (bx - cx_), res / 2 + sc_ * (by - cy_)
    capm = _head_mask(res, S_, w_, h_, bx, by, cx_, cy_, ax=1.0, ay=1.25, dy=0.2, fe=0.0) > 0.5
    mo = _head_region(sess, crop, fx, fy, w_ * sc_, h_ * sc_, capm)
    mg = _head_region(sess, gen, fx, fy, w_ * sc_, h_ * sc_, capm)
    u = np.maximum(mo, mg)
    kd = int(0.010 * res) | 1
    u = cv2.dilate(u, np.ones((kd, kd), np.uint8))
    mask = cv2.GaussianBlur(u, (0, 0), 0.006 * res)
    kr = int(0.06 * res) | 1
    gen = _color_match(gen, crop, cv2.dilate(u, np.ones((kr, kr), np.uint8)))
    inv = cv2.invertAffineTransform(M)
    gw = cv2.warpAffine(gen, inv, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT_101)
    mw = cv2.warpAffine(mask, inv, (W, H), flags=cv2.INTER_LINEAR, borderValue=0)[..., None].astype(np.float32)
    final = (img.astype(np.float32) * (1 - mw) + gw.astype(np.float32) * mw).clip(0, 255).astype(np.uint8)
    outp = f"{JOBS}/images/{jid}.png"
    cv2.imwrite(outp, final)
    jobs_vol.commit()
    uploaded = _put_file(outp, output_put_url, "image/png") if output_put_url else None
    return {"output": f"images/{jid}.png", "faces_found": len(faces), "size": [W, H], "res": res, "seconds": round(time.time() - t0, 1), "uploaded": uploaded,
            "model": "Qwen-Image-2.1 + bfs_head_v1.1_qwen_2.1"}


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
