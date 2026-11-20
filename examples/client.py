"""Example client for the Head Swap Studio API.

    python examples/client.py --api https://<workspace>--headswap-studio-api.us-east.modal.run \
        --image https://.../poster.png --face https://.../face.jpg --out result.png
"""
import argparse
import json
import time

import requests


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", required=True)
    ap.add_argument("--image", required=True, help="public URL of the poster")
    ap.add_argument("--face", required=True, help="public URL of the new face")
    ap.add_argument("--out", default="result.png")
    ap.add_argument("--engine", default="klein", choices=["klein", "bfs", "edit"])
    ap.add_argument("--best-of", type=int, default=4)
    ap.add_argument("--neck-depth", type=float, default=1.0)
    ap.add_argument("--max-rounds", type=int, default=2)
    a = ap.parse_args()

    body = {"image_url": a.image, "face_url": a.face, "engine": a.engine, "best_of": a.best_of,
            "neck_depth": a.neck_depth, "max_rounds": a.max_rounds}
    job = requests.post(f"{a.api}/headswap", json=body, timeout=120).json()
    print("job", job["job_id"])
    while True:
        time.sleep(10)
        s = requests.get(f"{a.api}/jobs/{job['job_id']}", timeout=120).json()
        if s["status"] != "running":
            break
    if s["status"] != "done":
        raise SystemExit(json.dumps(s, indent=1))
    r = s["result"]
    open(a.out, "wb").write(requests.get(r["download_url"], timeout=300).content)
    print("saved", a.out, "| passed quality gate:", r["passed_quality_gate"], "| quality:", r["quality"])


if __name__ == "__main__":
    main()
