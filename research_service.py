"""Separate Modal app for research head/face-swap models (does not touch bfs_service.py / bfs-headswap-api).

Endpoints are prepared for models whose code or weights are not public yet. Until a model is released, its endpoint
returns HTTP 501 with the release status and links, and no GPU is started (idle cost $0).

Deploy:  modal deploy research_service.py
URL:     https://<workspace>--headswap-research-api.us-east.modal.run

To add a model once it is released: set "released": True in MODELS, add a GPU class (see the commented template at the
bottom) and route the POST endpoint to it.
"""
import modal

app = modal.App("headswap-research")
web_image = modal.Image.debian_slim(python_version="3.11").pip_install("fastapi[standard]")

MODELS = {
    "livingswap": {
        "name": "LivingSwap (CVPR 2026)",
        "kind": "video face swap",
        "authors": "Zhejiang University (aim-uofa)",
        "released": False,
        "status": "No code or weights published: the project page and paper have no code link.",
        "links": ["https://aim-uofa.github.io/LivingSwap/", "https://arxiv.org/abs/2512.07951"],
        "last_checked": "2026-11-13",
    },
    "streamingswap": {
        "name": "StreamingSwap (2026, LivingSwap team follow-up)",
        "kind": "real-time video face swap",
        "authors": "Zhejiang University (aim-uofa)",
        "released": False,
        "status": "Repository contains only a README; inference code, weights and data are listed as not yet released.",
        "links": ["https://github.com/aim-uofa/StreamingSwap", "https://arxiv.org/abs/2610.26681"],
        "last_checked": "2026-11-13",
    },
    "gswap": {
        "name": "GSwap (2026)",
        "kind": "video head swap with a dynamic neural Gaussian field",
        "authors": "USTC 3DV",
        "released": False,
        "status": "No code or weights published: the project page and paper have no code link.",
        "links": ["https://ustc3dv.github.io/GSwap/", "https://arxiv.org/abs/2603.23168"],
        "last_checked": "2026-11-13",
    },
}


@app.function(image=web_image, cpu=0.25, memory=256)
@modal.asgi_app(label="headswap-research-api")
def web():
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import JSONResponse

    api = FastAPI(title="Head-swap research models",
                  description="Prepared endpoints for research models. Production head swaps run on bfs-headswap-api.")

    @api.get("/")
    def root():
        return {"app": "headswap-research", "models": {k: {"name": v["name"], "released": v["released"]} for k, v in MODELS.items()},
                "production_api": "bfs-headswap-api (unchanged)"}

    @api.get("/models")
    def models():
        return MODELS

    @api.get("/models/{model}")
    def model_info(model: str):
        if model not in MODELS:
            raise HTTPException(404, f"unknown model '{model}'; available: {', '.join(MODELS)}")
        return MODELS[model]

    def not_released(model: str):
        m = MODELS[model]
        return JSONResponse(status_code=501, content={
            "error": "model_not_released",
            "model": m["name"],
            "message": f"{m['name']} cannot run yet: {m['status']} The endpoint will work once the authors publish code and weights.",
            "links": m["links"],
            "last_checked": m["last_checked"],
        })

    @api.post("/livingswap")
    def livingswap(body: dict = None):
        return not_released("livingswap")

    @api.post("/streamingswap")
    def streamingswap(body: dict = None):
        return not_released("streamingswap")

    @api.post("/gswap")
    def gswap(body: dict = None):
        return not_released("gswap")

    return api


# ----------------------------------------------------------------------------- BFS-Best-Face-Swap-Video on LTX-2.5
# Video-native head swap: Alissonerdx/BFS-Best-Face-Swap-Video (ltx-2.5/head_swap_ltx25_r128_v1.1) on Lightricks
# LTX-2.5 22B distilled (int8). The author's drag-and-drop workflow is converted to ComfyUI API format at run time
# (KJNodes Set/Get resolved, UI-only nodes dropped); only the inputs below are changed:
#   guide video, reference image, fps / duration, resolution preset, and the auto-caption (TextGenerate with a
#   Qwen3-VL "heretic" model) is replaced by one fixed FACE description of the reference (it runs once per clip anyway).
# Licence: LTX-2.x Community License (free commercial use under $10M annual revenue). Model repo is gated -> needs the
# Hugging Face token from an account that accepted the licence, passed to download_ltx() at call time.
import os, json, time, subprocess, uuid, urllib.request, urllib.parse

MODELS_DIR = "/models"
research_models = modal.Volume.from_name("research-models", create_if_missing=True)
research_jobs = modal.Volume.from_name("research-jobs", create_if_missing=True)
WF_PATH = "/wf/bfs_video_ltx25.json"

ltx_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "ffmpeg", "libgl1", "libglib2.0-0", "fonts-freefont-ttf")
    .pip_install("torch==2.8.0", "torchvision==0.23.0", "torchaudio==2.8.0", index_url="https://download.pytorch.org/whl/cu128")
    .run_commands(
        "git clone --depth 1 https://github.com/comfyanonymous/ComfyUI /comfy",
        "cd /comfy && pip install -r requirements.txt",
        "cd /comfy/custom_nodes && git clone --depth 1 https://github.com/kijai/ComfyUI-KJNodes && pip install -r ComfyUI-KJNodes/requirements.txt",
        "cd /comfy/custom_nodes && git clone --depth 1 https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite && pip install -r ComfyUI-VideoHelperSuite/requirements.txt",
        "cd /comfy/custom_nodes && git clone --depth 1 https://github.com/alisson-anjos/ComfyUI-BFSNodes && (pip install -r ComfyUI-BFSNodes/requirements.txt || true)",
        "pip install huggingface_hub hf_transfer",
    )
    .add_local_file(os.path.join(os.path.dirname(os.path.abspath(__file__)), "workflows_bfs_video_ltx25_v1.1.json"), WF_PATH)
)

LTX_FILES = [  # (repo, path in repo, destination under /models)
    ("Lightricks/LTX-2.5", "diffusion_models/ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors", "diffusion_models/ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors"),
    ("Lightricks/LTX-2.5", "text_encoders/gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors", "text_encoders/gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors"),
    ("Lightricks/LTX-2.5", "vae/ltx-2.5-video-vae-bf16.safetensors", "vae/ltx-2.5-video-vae-bf16.safetensors"),
    ("Lightricks/LTX-2.5", "vae/ltx-2.5-audio-vae-bf16.safetensors", "vae/ltx-2.5-audio-vae-bf16.safetensors"),
    ("Lightricks/LTX-2.5", "latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors", "latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors"),
    ("Alissonerdx/BFS-Best-Face-Swap-Video", "ltx-2.5/head_swap_ltx25_r128_v1.1.safetensors", "loras/head_swap_ltx25_v1_1_r128.safetensors"),
]


@app.function(image=ltx_image, volumes={MODELS_DIR: research_models}, timeout=7200)
def download_ltx(hf_token: str = ""):
    """hf_token: passed at call time (not stored); needs access to the gated Lightricks/LTX-2.5 repo"""
    import shutil
    from huggingface_hub import hf_hub_download
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    got = []
    for repo, path, dest in LTX_FILES:
        out = os.path.join(MODELS_DIR, dest)
        if os.path.exists(out):
            got.append(("have", dest))
            continue
        p = hf_hub_download(repo, path, token=hf_token or os.environ.get("HF_TOKEN"))
        os.makedirs(os.path.dirname(out), exist_ok=True)
        shutil.copyfile(p, out)
        os.remove(p)
        got.append(("got", dest, os.path.getsize(out)))
    research_models.commit()
    return got


UI_ONLY = {"SetNode", "GetNode", "Reroute", "Note", "MarkdownNote", "PrimitiveNode"}


def _ui_to_api(wf: dict, object_info: dict) -> dict:
    """ComfyUI UI workflow -> API prompt. Resolves KJNodes Set/Get, reroutes and bypassed nodes."""
    nodes = {n["id"]: n for n in wf["nodes"]}
    links = {l[0]: l for l in wf["links"]}            # id, from_node, from_slot, to_node, to_slot, type
    setters = {n["widgets_values"][0]: n for n in wf["nodes"] if n["type"] == "SetNode"}

    def resolve(link_id):
        _, src, slot, _, _, typ = links[link_id]
        n = nodes[src]
        if n["type"] == "GetNode":
            s = setters[n["widgets_values"][0]]
            return resolve(s["inputs"][0]["link"])
        if n["type"] in ("Reroute", "SetNode"):
            return resolve(n["inputs"][0]["link"])
        if n.get("mode") == 4:                         # bypassed: pass the first input of the same type through
            for i in n.get("inputs", []):
                if i.get("link") and i.get("type") == typ:
                    return resolve(i["link"])
        return [str(src), slot]

    api = {}
    for n in wf["nodes"]:
        if n["type"] in UI_ONLY or n.get("mode") in (2, 4):
            continue
        info = object_info.get(n["type"])
        if info is None:
            raise RuntimeError(f"node type not installed: {n['type']}")
        order = info.get("input_order") or {"required": list(info["input"].get("required", {})), "optional": list(info["input"].get("optional", {}))}
        specs = {**info["input"].get("required", {}), **info["input"].get("optional", {})}
        linked = {i["name"]: i["link"] for i in n.get("inputs", []) if i.get("link") is not None}
        ui_widgets = {i["name"] for i in n.get("inputs", []) if i.get("widget")}
        wv = n.get("widgets_values")
        inputs = {}
        if isinstance(wv, dict):
            for k, v in wv.items():
                if k in specs:
                    inputs[k] = v
        else:
            wv = list(wv or [])
            k = 0
            for name in order.get("required", []) + order.get("optional", []):
                spec = specs.get(name)
                if spec is None:
                    continue
                t = spec[0]
                opts = spec[1] if len(spec) > 1 and isinstance(spec[1], dict) else {}
                is_widget = isinstance(t, list) or t in ("INT", "FLOAT", "STRING", "BOOLEAN", "COMBO") or opts.get("multiline") is not None
                if opts.get("forceInput") or opts.get("rawLink"):
                    is_widget = False
                if name in ui_widgets:                   # the saved workflow had a widget here (newer node defs may
                    is_widget = True                     # mark it input-only), so it still owns a widgets_values slot
                if not is_widget:
                    continue
                if k < len(wv):
                    inputs[name] = wv[k]
                k += 1
                if t in ("INT", "FLOAT") and (opts.get("control_after_generate") or name in ("seed", "noise_seed")):
                    k += 1                               # skip the "fixed / randomize" control value
        for name, lid in linked.items():
            if name in specs or "." in name:          # dotted names = dynamic/autogrow inputs (e.g. values.a)
                inputs[name] = resolve(lid)
        api[str(n["id"])] = {"class_type": n["type"], "inputs": inputs}
    return api


FACE_TEXT = "head_swap:\nFACE: Male, fair skin, approximately 25-35 years old, light blue-grey eyes, short dark brown textured hair swept up."


@app.cls(image=ltx_image, gpu="L40S", volumes={MODELS_DIR: research_models}, timeout=3600, scaledown_window=15)
class LTXHeadSwapWorker:
    @modal.enter()
    def boot(self):
        for sub in ["diffusion_models", "text_encoders", "vae", "loras", "latent_upscale_models"]:
            d = f"/comfy/models/{sub}"
            subprocess.run(["rm", "-rf", d])
            os.makedirs(f"{MODELS_DIR}/{sub}", exist_ok=True)
            os.symlink(f"{MODELS_DIR}/{sub}", d)
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

    def _obj_info(self):
        return json.load(urllib.request.urlopen("http://127.0.0.1:8188/object_info"))

    @modal.method()
    def convert(self) -> dict:
        """debug: the converted API prompt + node definitions of the nodes it uses (no generation)"""
        info = self._obj_info()
        api = _ui_to_api(json.load(open(WF_PATH)), info)
        return {"api": api, "defs": {v["class_type"]: info[v["class_type"]].get("input_order") for v in api.values()}}

    @modal.method()
    def run(self, guide_mp4: bytes, ref_png: bytes, fps: int = 24, seconds: int = 4, face_text: str = FACE_TEXT,
            seed: int = 42, lora_strength: float = 0.8, aspect_hint: str = "9:16", refine_sigmas: str = "") -> dict:
        """refine_sigmas: partial-strength refine. The first pass starts from the VAE-encoded guide (e.g. a finished BFS
        swap) instead of empty noise, and samples only the given (lower) sigmas, so the guide's identity and expressions
        are kept and the video model mainly removes frame-to-frame shimmer."""
        os.makedirs("/comfy/input", exist_ok=True)
        open("/comfy/input/guide.mp4", "wb").write(guide_mp4)
        open("/comfy/input/ref.png", "wb").write(ref_png)
        info = self._obj_info()
        wf = json.load(open(WF_PATH))
        api = _ui_to_api(wf, info)
        # inputs
        api["460"]["inputs"]["video"] = "guide.mp4"
        api["461"]["inputs"]["image"] = "ref.png"
        api["425"]["inputs"]["value"] = seconds
        api["430"]["inputs"]["value"] = fps
        api["610"]["inputs"]["strength_model"] = lora_strength
        api["432"]["inputs"]["noise_seed"] = seed
        res_note = None
        if "403" in api:   # resolution preset: pick the option matching the guide's aspect (keep the workflow's value otherwise)
            spec = info.get("ResolutionSelector", {}).get("input", {})
            allin = {**spec.get("required", {}), **spec.get("optional", {})}
            for name, v in allin.items():
                opts = v[0] if isinstance(v[0], list) else (v[1].get("options") if len(v) > 1 and isinstance(v[1], dict) else None)
                if opts and any(":" in str(o) for o in opts):
                    pick = next((o for o in opts if str(o).startswith(aspect_hint)), None)
                    if pick:
                        api["403"]["inputs"][name] = pick
                    res_note = {"input": name, "picked": pick, "options": [str(o) for o in opts][:20]}
                    break
        # fixed FACE description instead of the Qwen3-VL auto caption; drop caption + preview/compare outputs
        api["448"]["inputs"]["text"] = face_text
        api["75"]["inputs"].update({"format": "auto", "codec": "auto"})
        for nid in ("447", "445", "607", "567", "491", "477", "480", "481", "482", "483", "484", "600", "476"):
            api.pop(nid, None)
        if refine_sigmas:
            # 462 = guide resized to the first-pass size, 453 = LTX video VAE, 456 = the empty video latent it replaces
            api["900"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["462", 0], "vae": ["453", 0]}}
            for node in api.values():
                for k, v in node["inputs"].items():
                    if v == ["456", 0]:
                        node["inputs"][k] = ["900", 0]
            api.pop("456", None)
            api["471"]["inputs"]["sigmas"] = refine_sigmas
        req = urllib.request.Request("http://127.0.0.1:8188/prompt", headers={"Content-Type": "application/json"},
                                     data=json.dumps({"prompt": api}).encode())
        t0 = time.time()
        try:
            pid = json.load(urllib.request.urlopen(req))["prompt_id"]
        except urllib.error.HTTPError as e:
            raise RuntimeError(e.read().decode()[:6000])
        while True:
            h = json.load(urllib.request.urlopen(f"http://127.0.0.1:8188/history/{pid}"))
            if pid in h:
                break
            time.sleep(2)
        if h[pid]["status"].get("status_str") != "success":
            raise RuntimeError("comfy failed: " + json.dumps(h[pid]["status"])[:4000] + "\nLOG:" + open("/tmp/comfy.log").read()[-3000:])
        out = h[pid]["outputs"].get("75", {})
        files = out.get("images") or out.get("videos") or out.get("gifs") or []
        if not files:
            raise RuntimeError("no video output: " + json.dumps(h[pid]["outputs"])[:2000])
        f = files[0]
        q = urllib.parse.urlencode({"filename": f["filename"], "subfolder": f.get("subfolder", ""), "type": f.get("type", "output")})
        data = urllib.request.urlopen(f"http://127.0.0.1:8188/view?{q}").read()
        return {"video": data, "seconds": round(time.time() - t0, 1), "filename": f["filename"], "resolution": res_note}


# ----------------------------------------------------------------------------- head paste-back (CPU)
seg_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libgl1", "libglib2.0-0")
    .pip_install("numpy<2", "opencv-python-headless==4.10.0.84", "onnxruntime==1.20.1")
    .run_commands("mkdir -p /seg", "python -c \"import urllib.request as u; u.urlretrieve('https://huggingface.co/jonathandinu/face-parsing/resolve/main/onnx/model_quantized.onnx', '/seg/model.onnx')\"")
)
HEAD_CLASSES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17]   # skin..lips, hair, hat, earring, neck (not cloth, not hands)


@app.function(image=seg_image, cpu=8, memory=16384, timeout=3600)
def compose_head(orig_mp4: bytes, gen_mp4: bytes, boxes: list, cm_radius: int = 3) -> bytes:
    """Paste the generated head onto the original frames: face-parsing mask (union of original and generated
    head + hair, so hands / food / background stay original), colour-matched to the original (LAB transfer measured
    on a ring around the head, smoothed over +-cm_radius frames). boxes: per-frame face box [x0,y0,x1,y1] or None."""
    import cv2, numpy as np, onnxruntime as ort, tempfile
    sess = ort.InferenceSession("/seg/model.onnx", providers=["CPUExecutionProvider"])

    def frames(b):
        p = tempfile.mktemp(suffix=".mp4"); open(p, "wb").write(b)
        c = cv2.VideoCapture(p); out = []
        while True:
            ok, f = c.read()
            if not ok:
                break
            out.append(f)
        return out
    O, G = frames(orig_mp4), frames(gen_mp4)
    n = min(len(O), len(G))
    H, W = O[0].shape[:2]
    idx = [i for i in range(n) if boxes[i]]
    Bx = {i: np.array(boxes[i], float) for i in idx}

    def sbox(i):
        ks = [j for j in range(i - 4, i + 5) if j in Bx]
        w = np.array([np.exp(-0.5 * ((j - i) / 2.0) ** 2) for j in ks])
        return (np.array([Bx[j] for j in ks]) * w[:, None]).sum(0) / w.sum()

    def labels(img):
        x = cv2.resize(img, (512, 512))[..., ::-1].astype(np.float32) / 255.0
        x = ((x - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]).transpose(2, 0, 1)[None].astype(np.float32)
        lg = sess.run(None, {sess.get_inputs()[0].name: x})[0][0]
        return np.stack([cv2.resize(c, (512, 512)) for c in lg]).argmax(0)

    def head_mask(img, cap):
        lab = labels(img)
        m = (np.isin(lab, HEAD_CLASSES) & cap).astype(np.uint8)
        nlab, cc = cv2.connectedComponents(m, connectivity=8)
        core = np.zeros((512, 512), np.uint8)
        cv2.ellipse(core, (256, int(256 + 0.12 * 512 / 2.6)), (40, 50), 0, 0, 360, 1, -1)
        keep = np.unique(cc[(core > 0) & (cc > 0)])
        return np.isin(cc, keep).astype(np.float32) if len(keep) else m.astype(np.float32)

    geo, masks, params = {}, {}, {}
    for i in idx:
        x0, y0, x1, y1 = sbox(i)
        w, h = x1 - x0, y1 - y0
        cx, cy, S = (x0 + x1) / 2, (y0 + y1) / 2 - 0.12 * h, 2.6 * max(h, w * 1.15)
        sc = 512 / S
        M = np.array([[sc, 0, 256 - sc * cx], [0, sc, 256 - sc * cy]], np.float32)
        co = cv2.warpAffine(O[i], M, (512, 512), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_REFLECT)
        cg = cv2.warpAffine(G[i], M, (512, 512), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_REFLECT)
        cap = np.zeros((512, 512), np.uint8)
        cv2.ellipse(cap, (256, int(256 + 0.12 * h * sc)), (int(1.0 * w * sc), int(1.3 * h * sc)), 0, 0, 360, 1, -1)
        u = np.maximum(head_mask(co, cap > 0), head_mask(cg, cap > 0))
        u = cv2.dilate(u, np.ones((7, 7), np.uint8))
        ring = (cv2.dilate(u, np.ones((31, 31), np.uint8)) > 0) & (u < 0.5)
        lo, lg = cv2.cvtColor(co, cv2.COLOR_BGR2LAB).astype(np.float32), cv2.cvtColor(cg, cv2.COLOR_BGR2LAB).astype(np.float32)
        p = np.zeros((3, 2), np.float32)
        p[:, 0] = 1.0
        if ring.sum() > 300:
            for c in range(3):
                gm, gs = lg[..., c][ring].mean(), lg[..., c][ring].std() + 1e-3
                om, os_ = lo[..., c][ring].mean(), lo[..., c][ring].std() + 1e-3
                a = float(np.clip(os_ / gs, 0.85, 1.2))
                p[c] = (a, om - gm * a)
        params[i] = p
        geo[i] = M
        masks[i] = cv2.GaussianBlur(u, (0, 0), 3.0)
    out = list(O[:n])
    for i in idx:
        ks = [j for j in range(i - cm_radius, i + cm_radius + 1) if j in params]
        p = np.mean([params[j] for j in ks], 0)
        g = cv2.cvtColor(G[i], cv2.COLOR_BGR2LAB).astype(np.float32)
        for c in range(3):
            g[..., c] = g[..., c] * p[c, 0] + p[c, 1]
        g = cv2.cvtColor(np.clip(g, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR).astype(np.float32)
        mw = cv2.warpAffine(masks[i], cv2.invertAffineTransform(geo[i]), (W, H), flags=cv2.INTER_LINEAR, borderValue=0)[..., None]
        out[i] = (O[i].astype(np.float32) * (1 - mw) + g * mw).clip(0, 255).astype(np.uint8)
    tmp = tempfile.mktemp(suffix=".mp4")
    pr = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", "25", "-i", "-",
                           "-c:v", "libx264", "-crf", "14", "-pix_fmt", "yuv420p", tmp], stdin=subprocess.PIPE)
    for f in out:
        pr.stdin.write(f.tobytes())
    pr.stdin.close()
    pr.wait()
    return open(tmp, "rb").read()


# Template for a released model (kept commented so nothing is built or billed until it exists):
#
# gpu_image = modal.Image.debian_slim(python_version="3.11").apt_install("git", "ffmpeg").pip_install(...)
# models_vol = modal.Volume.from_name("research-models", create_if_missing=True)
# jobs_vol = modal.Volume.from_name("research-jobs", create_if_missing=True)
#
# @app.cls(image=gpu_image, gpu="L40S", volumes={"/models": models_vol, "/jobs": jobs_vol}, scaledown_window=15, timeout=3600)
# class LivingSwapWorker:
#     @modal.enter()
#     def load(self): ...
#     @modal.method()
#     def swap(self, video_url: str, source_face_url: str) -> str: ...
