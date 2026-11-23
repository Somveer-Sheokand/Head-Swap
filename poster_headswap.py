"""
Head swap one poster with several faces, ashymer/modal-blender style:
structured poster-specific prompt, high-resolution generation (up to 2048), turbo and/or base, several seeds;
the API puts the result back at the poster's original size with only the head changed.

  python poster_headswap.py POSTER PROMPT_FILE OUT_DIR --faces a.jpg b.jpg --models turbo base --seeds 7 42
"""

import argparse
import concurrent.futures as cf
import os
import time

import requests
from PIL import Image

API = os.environ.get("QWEN_API_URL", "https://ashish-kumar--qwen-image-2-1-api.us-east.modal.run")


def gen_size(poster, long_side=2048):
    w, h = Image.open(poster).size
    s = long_side / max(w, h)  # generate at the model's max resolution for face detail
    return f"{round(w * s / 16) * 16}x{round(h * s / 16) * 16}"


def run(poster, face, prompt, model, seed, out_dir, long_side=2048):
    with open(poster, "rb") as t, open(face, "rb") as f:
        r = requests.post(
            f"{API}/v1/headswap",
            files={"target_image": t, "source_image": f},
            data={"prompt": prompt, "model": model, "seed": seed, "size": gen_size(poster, long_side), "preserve_original": "true"},
            timeout=180,
        )
    r.raise_for_status()
    job = r.json()["job_id"]
    t0 = time.time()
    while True:
        st = requests.get(f"{API}/jobs/{job}", timeout=60).json()
        if st["status"] != "pending":
            break
        time.sleep(4)
    if st["status"] != "completed":
        return f"FAIL {face} {model} s{seed}: {st.get('error')}"
    name = os.path.join(out_dir, f"{os.path.splitext(os.path.basename(face))[0]}__{model}__s{seed}.png")
    with open(name, "wb") as fh:
        fh.write(requests.get(st["download_url"], timeout=120).content)
    return f"ok {os.path.basename(name)} ({job}, {time.time() - t0:.0f}s)"


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("poster")
    p.add_argument("prompt_file")
    p.add_argument("out_dir")
    p.add_argument("--faces", nargs="+", required=True)
    p.add_argument("--models", nargs="+", default=["turbo", "base"])
    p.add_argument("--seeds", nargs="+", type=int, default=[7, 42])
    p.add_argument("--long-side", type=int, default=2048, help="generation size; base+2 refs needs <=1600 on H100")
    a = p.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    prompt = open(a.prompt_file, encoding="utf-8").read().strip()
    jobs = [(f, m, s) for f in a.faces for m in a.models for s in a.seeds]
    with cf.ThreadPoolExecutor(len(jobs)) as ex:
        for line in ex.map(lambda j: run(a.poster, j[0], prompt, j[1], j[2], a.out_dir, a.long_side), jobs):
            print(line)
