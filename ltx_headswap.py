"""Video head swap with BFS-Best-Face-Swap-Video on LTX-2.5 (runs on the headswap-research Modal app).

Pipeline (each step fixes an issue found in the first LTX tests):
  1. Split the person's frames into camera shots (real cuts)           -> no identity jump at cuts
  2. One fixed crop per shot around the person (zoom small faces)      -> wide shots get swapped too
  3. Guide clip = original frames written 1:1 at 24 fps (no resampling) -> no repeated / skipped frames
  4. LTX-2.5 + BFS-Video LoRA per shot, one after another in one GPU container (models load once)
  5. Generated crops placed back 1:1, then head-only paste (face parsing) + colour match -> text, logos,
     hands, food and background stay original
  6. Original audio added

Inputs: a preprocessed template (video.mp4 + track.json from bfs-headswap-api /preprocess, or the same format),
a reference head photo and a short FACE description.

Usage:
  python ltx_headswap.py --video video.mp4 --track track.json --ref head.png \
      --face "Male, medium-brown skin, 40-50 years old, dark brown eyes, very short buzz-cut hair, light stubble." \
      --out result.mp4 [--lora 0.9] [--dry-run]

Deploy first:  modal deploy research_service.py  (and once: download_ltx with a Hugging Face token that accepted the
LTX-2.x licence). Cost (L40S): about $0.15-0.25 per shot of 2-3 s plus one model load; check `--dry-run` first.
"""
import argparse, json, math, os, subprocess, tempfile, time

import cv2
import numpy as np


def read_frames(path):
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    return frames, cap.get(cv2.CAP_PROP_FPS) or 25.0


def find_cuts(frames, thresh=0.55):
    """Hard cuts: colour-histogram correlation between consecutive frames drops (robust to fast motion)."""
    def hist(f):
        hsv = cv2.cvtColor(cv2.resize(f, (180, 320)), cv2.COLOR_BGR2HSV)
        h = cv2.calcHist([hsv], [0, 1], None, [32, 16], [0, 180, 0, 256])
        return cv2.normalize(h, h).flatten()
    hs = [hist(f) for f in frames]
    small = [cv2.resize(f, (180, 320)).astype(np.int16) for f in frames]
    d = [0.0] + [float(np.abs(small[i] - small[i - 1]).mean()) for i in range(1, len(frames))] + [0.0]
    cuts = set()
    for i in range(1, len(frames)):
        if cv2.compareHist(hs[i - 1], hs[i], cv2.HISTCMP_CORREL) < thresh:
            cuts.add(i)                                   # colours change: different scene
        elif d[i] > 20 and d[i] > 3 * max(d[i - 1], d[i + 1]):
            cuts.add(i)                                   # one-frame spike: cut within the same scene (motion is gradual)
    return cuts


def find_shots(frames, track, min_len=12):
    """Runs of frames that contain the person, split at hard cuts; pieces shorter than min_len are merged into the
    neighbouring piece (so no frame of the person is left unswapped)."""
    cuts = find_cuts(frames)
    runs, cur = [], []
    for i in range(len(frames)):
        has = bool(track[i]["box"])
        if cur and (not has or i in cuts):
            runs.append(cur)
            cur = []
        if has:
            cur.append(i)
    if cur:
        runs.append(cur)
    merged = []
    for r in runs:
        if merged and (len(r) < min_len or len(merged[-1]) < min_len) and r[0] == merged[-1][-1] + 1:
            merged[-1] = merged[-1] + r
        else:
            merged.append(r)
    return [(r[0], r[-1]) for r in merged if len(r) >= 5]


def shot_crop(track, a, b, W, H, zoom=5.0):
    """Fixed 9:16 crop around the person for the whole shot; small faces get zoomed in."""
    bx = np.array([track[i]["box"] for i in range(a, b + 1) if track[i]["box"]], float)
    fh = float(np.median(bx[:, 3] - bx[:, 1]))
    ux0, uy0, ux1, uy1 = bx[:, 0].min(), bx[:, 1].min(), bx[:, 2].max(), bx[:, 3].max()
    h = float(np.clip(max(zoom * fh, (uy1 - uy0) * 2.2, (ux1 - ux0) * 16 / 9 * 1.3), 640, H))
    w = h * 9 / 16
    if w > W:
        w, h = W, W * 16 / 9
    cx, cy = (ux0 + ux1) / 2, (uy0 + uy1) / 2 + 0.25 * fh
    x0 = int(np.clip(cx - w / 2, 0, W - w))
    y0 = int(np.clip(cy - h / 2, 0, H - h))
    return [x0, y0, int(round(w)), int(round(h))], fh


def write_guide(frames, a, b, crop, secs, path):
    """Original frames 1:1, labelled 24 fps (LTX rate) - no frame is dropped or repeated; padded by holding the
    last frame up to the 24*secs+1 length LTX needs."""
    x0, y0, w, h = crop
    seq = [frames[i][y0:y0 + h, x0:x0 + w] for i in range(a, b + 1)]
    seq += [seq[-1]] * (secs * 24 + 1 - len(seq))
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", "24",
                          "-i", "-", "-vf", "scale=576:1024:flags=lanczos", "-c:v", "libx264", "-crf", "12",
                          "-pix_fmt", "yuv420p", path], stdin=subprocess.PIPE)
    for f in seq:
        p.stdin.write(np.ascontiguousarray(f).tobytes())
    p.stdin.close()
    p.wait()


def encode(frames, fps, path):
    H, W = frames[0].shape[:2]
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(fps),
                          "-i", "-", "-c:v", "libx264", "-crf", "12", "-pix_fmt", "yuv420p", path], stdin=subprocess.PIPE)
    for f in frames:
        p.stdin.write(f.tobytes())
    p.stdin.close()
    p.wait()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True)
    ap.add_argument("--track", required=True, help="track.json with per-frame head boxes")
    ap.add_argument("--ref", required=True, help="reference photo, cropped to head and shoulders")
    ap.add_argument("--face", required=True, help="short identity description of the reference")
    ap.add_argument("--out", required=True)
    ap.add_argument("--lora", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--app", default="headswap-research")
    ap.add_argument("--dry-run", action="store_true", help="only print shots, crops and the cost estimate")
    args = ap.parse_args()

    frames, fps = read_frames(args.video)
    track = json.load(open(args.track))["track"]
    H, W = frames[0].shape[:2]
    shots = find_shots(frames, track)
    plan = []
    for k, (a, b) in enumerate(shots):
        crop, fh = shot_crop(track, a, b, W, H)
        n = b - a + 1
        plan.append({"shot": k, "a": a, "b": b, "frames": n, "seconds": max(1, math.ceil((n - 1) / 24)), "crop": crop,
                     "face_px_in_model": int(fh * 1024 / crop[3])})
    gpu_s = 150 + sum(70 * p["seconds"] + 30 for p in plan)          # one model load + per-shot sampling (L40S)
    print(json.dumps(plan, indent=1))
    print(f"{len(plan)} shots, estimated GPU time {gpu_s / 60:.1f} min, cost about ${gpu_s * 1.95 / 3600:.2f} (L40S)")
    if args.dry_run:
        return

    import modal
    worker = modal.Cls.from_name(args.app, "LTXHeadSwapWorker")()
    compose = modal.Function.from_name(args.app, "compose_head")
    ref = open(args.ref, "rb").read()
    face = args.face if args.face.startswith("head_swap:") else "head_swap:\nFACE: " + args.face
    tmp = tempfile.mkdtemp(prefix="ltxswap_")
    gen = [f.copy() for f in frames]
    boxes = [None] * len(frames)
    t0 = time.time()
    for p in plan:                       # sequential -> the same warm GPU container, models load once
        guide = os.path.join(tmp, f"guide{p['shot']}.mp4")
        write_guide(frames, p["a"], p["b"], p["crop"], p["seconds"], guide)
        r = worker.run.remote(open(guide, "rb").read(), ref, fps=24, seconds=p["seconds"], seed=args.seed,
                              lora_strength=args.lora, face_text=face, aspect_hint="9:16")
        out = os.path.join(tmp, f"shot{p['shot']}.mp4")
        open(out, "wb").write(r["video"])
        x0, y0, cw, ch = p["crop"]
        cap, k = cv2.VideoCapture(out), 0
        while k < p["frames"]:          # 1:1 frame mapping back to the original frames
            ok, f = cap.read()
            if not ok:
                break
            i = p["a"] + k
            gen[i][y0:y0 + ch, x0:x0 + cw] = cv2.resize(f, (cw, ch), interpolation=cv2.INTER_AREA if ch < f.shape[0] else cv2.INTER_LANCZOS4)
            boxes[i] = track[i]["box"]
            k += 1
        print(f"shot {p['shot']} ({p['a']}-{p['b']}) done in {r['seconds']:.0f}s, {k} frames mapped", flush=True)
    gen_path = os.path.join(tmp, "gen_full.mp4")
    encode(gen, fps, gen_path)
    head = compose.remote(open(args.video, "rb").read(), open(gen_path, "rb").read(), boxes)
    head_path = os.path.join(tmp, "head.mp4")
    open(head_path, "wb").write(head)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", head_path, "-i", args.video, "-map", "0:v", "-map", "1:a?",
                    "-c:v", "copy", "-c:a", "aac", "-b:a", "160k", "-shortest", args.out], check=True)
    print(f"saved {args.out} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
