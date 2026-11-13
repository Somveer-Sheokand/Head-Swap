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
