"""
Call the head-swap API exactly like the shared curl (POST /jobs -> GET status_url -> download_url).

  python run_jobs.py TEMPLATE_URL FACE_URL --name NAME --seeds 42 123 [--prompt-file p.txt] [--out DIR]

Without --prompt-file the API's default prompt is used (fetched and sent explicitly so it is logged).
"""

import argparse
import concurrent.futures as cf
import json
import os
import time

import requests

API = os.environ.get("QWEN_API_URL", "https://ashish-kumar--qwen-image-2-1-api.us-east.modal.run")


def _retry(fn, *a, **kw):
    for _ in range(30):
        try:
            return fn(*a, **kw)
        except requests.exceptions.ConnectionError:
            time.sleep(10)
    raise RuntimeError("network unavailable")


def run(template_url, face_url, prompt, seed, name, out):
    body = {"image_url": template_url, "source_face_url": face_url, "prompt": prompt, "seed": seed}
    gen = _retry(requests.post, f"{API}/jobs", json=body, timeout=180).json()
    while True:
        st = _retry(requests.get, gen["status_url"], timeout=60).json()
        if st["status"] != "pending":
            break
        time.sleep(5)
    stem = f"{name}__s{seed}"
    if st["status"] == "completed":
        with open(os.path.join(out, stem + ".png"), "wb") as f:
            f.write(_retry(requests.get, st["download_url"], timeout=120).content)
    with open(os.path.join(out, stem + ".json"), "w") as f:
        json.dump({"request": body, "generation_response": gen, "status_response": st}, f, indent=2)
    return stem, st["status"]


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("template_url")
    p.add_argument("face_url")
    p.add_argument("--name", required=True)
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 123])
    p.add_argument("--prompt-file")
    p.add_argument("--out", default="output/api_demo")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    prompt = (
        open(a.prompt_file, encoding="utf-8").read().strip()
        if a.prompt_file
        else _retry(requests.get, f"{API}/v1/headswap/prompt", timeout=60).json()["default_prompt"]
    )
    with cf.ThreadPoolExecutor(len(a.seeds)) as ex:
        for r in ex.map(lambda s: run(a.template_url, a.face_url, prompt, s, a.name, a.out), a.seeds):
            print(*r)
