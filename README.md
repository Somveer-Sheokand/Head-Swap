# Head-Swap (BFS on Modal)

A hosted head-swap API for **images and videos**, running on [Modal](https://modal.com).

- **Model:** [Qwen Image 2.1](https://huggingface.co/Comfy-Org/Qwen-Image-2.1) (int8) + the
  [BFS head-swap LoRA](https://huggingface.co/Alissonerdx/BFS-Best-Face-Swap) `bfs_head_v1.1_qwen_2.1`
  (plus Pruna's 8-step speed LoRA), run through ComfyUI on an NVIDIA L40S.
- **Pipeline (per frame):** detect the target head (InsightFace) -> crop around it -> BFS head swap on the crop ->
  hair/face segmentation of the old and the new head -> paste back the union of both outlines with a feathered mask ->
  colour-match -> light temporal smoothing across frames -> re-encode with the original audio.

## Files

| File | What it is |
|---|---|
| `bfs_service.py` | The whole Modal app: GPU worker (ComfyUI), preprocessing, image swap, video swap, HTTP API. |
| `download_models.py` | One-off job that downloads the model files into a Modal Volume. |

## Setup

```bash
pip install modal
modal token set --token-id <id> --token-secret <secret>      # your own Modal credentials

modal run download_models.py::download                       # ~18 GB into the "bfs-models" volume (once)
modal secret create bfs-api-token API_TOKEN=<any-random-string>
modal deploy bfs_service.py
```

`modal deploy` prints the public URL, e.g. `https://<workspace>--bfs-headswap-api.us-east.modal.run`.

> **Authentication is OFF by default** (`REQUIRE_AUTH = False` inside `web()` in `bfs_service.py`). Anyone with the URL can
> start jobs that spend your Modal credit and download results. Set it to `True` to require
> `Authorization: Bearer <API_TOKEN>`.

## API

All calls that start work return `{"job_id": "...", "status": "running"}` immediately; poll `GET /jobs/{job_id}`.
A finished job returns `result.output` and a ready-to-use `result.download_url`.

### Image
```bash
curl -X POST "$BASE/swap-image" -H "Content-Type: application/json" \
  -d '{"image_url":"https://.../photo.jpg","source_face_url":"https://.../head.jpg","face_index":1}'
curl "$BASE/jobs/<job_id>"            # -> result.download_url
```
`face_index` (0 = left-most) or `target_face_url` (crop of the person) selects whom to replace; default = largest face.
Optional: `quality`, `seed`, `output_put_url` (pre-signed PUT URL), `max_cost_usd`.

### Video
```bash
curl -X POST "$BASE/preprocess" -H "Content-Type: application/json" \
  -d '{"template_id":"my-video-1","video_url":"https://.../video.mp4","target_face_url":"https://.../actor-face.png"}'

curl -X POST "$BASE/swap" -H "Content-Type: application/json" \
  -d '{"template_id":"my-video-1","source_face_url":"https://.../head.jpg"}'

curl "$BASE/jobs/<job_id>"            # -> result.download_url
```
`/swap` options: `start_frame`, `max_frames` (process only part of the video), `max_cost_usd` (default 3),
`output_put_url`, `dry_run` (only returns frame count + cost estimate, no GPU), and `quality`:

| `quality` | What it does |
|---|---|
| `"fast"` (**default**) | up to 832 px (smaller for small / far-away heads, min 640 px), every 2nd frame generated, frames in between carried by motion (frames where that fails are generated too) |
| `"best"` | 1024 px, every head frame generated |

`POST /swap-video` does preprocess + swap in one call (`video_url`, `target_face_url`, `source_face_url`).
`/swap-image` uses the same `quality` values (default `"fast"`).

### How a video job runs
1. GPU stage once (`swap2`, `gpu_only`): heads are generated on 1-4 L40S containers (>= ~40 frames each, fewer cold starts).
2. Blending is split into 24-frame slices (6 frames overlap each side) and run in parallel on up to 12 small CPU
   containers (`blend_part`).
3. `swap_job` joins the slices with ffmpeg, adds the original audio and deletes the temporary files.

### Idle cost and storage
- Nothing stays running: GPU scales to zero 15 s after a job, CPU/API containers also scale to zero -> **$0 compute when idle**.
- Temporary frame folders are deleted after each job; outputs/uploads older than **14 days** are deleted automatically
  (`KEEP_DAYS`). Preprocessed templates (`video.mp4` + `track.json`) are kept.

## Time and cost (estimates, Modal L40S)

| Job | Time | Approx. cost |
|---|---|---|
| Image (1080x1920) | ~1-2 min (mostly cold start) | ~$0.10-0.20 |
| Video, 8 s, `fast` | ~8-10 min | ~$1.5-2 |
| Video, 8 s, `best` | ~18-20 min | ~$3.5-4 |

Check real usage with `modal billing report --for "this month"`; estimates have been low in the past, so the
built-in cost check (`max_cost_usd`) now multiplies the per-frame estimate by `EST_FACTOR = 2.0`.
Use `"dry_run": true` on `/swap` to see the frame count, generation size (`res`) and cost estimate without using a GPU.

### Optional speed-ups (built in, OFF until tested)
Two switches at the top of `bfs_service.py`. Each needs one short paid test (an image or a 2-second sample) after
switching it on and redeploying; switch it back off if the test fails.

| Switch | What it does | Expected gain |
|---|---|---|
| `USE_SNAPSHOT = True` | Modal memory + GPU snapshot of the loaded ComfyUI (models warmed up before the snapshot) | GPU cold start ~60-110 s -> ~10-20 s; images ~50 % cheaper |
| `USE_SAGE = True` | SageAttention in ComfyUI (`--use-sage-attention`, adds `sageattention` to the GPU image -> image rebuild) | ~15-30 % less GPU time per frame |

A further option is fewer sampling steps (`"steps": 6` in the `/swap` or `/swap-image` body; default 8). Compare
quality side by side before making it a default.
Very large images (tens of megapixels) are slower because blending runs on the full image.

## Known limitations

- Each frame's head is generated independently, so some frame-to-frame shimmer remains.
- In some side-profile frames a thin sliver of the old hair can remain at the very edge of the picture.
- Very fast zooms / head turns can warp slightly in `fast` mode; use `"quality":"best"` for such footage.
- The parallel join step (`swap_job`) was verified in parts only (slices blended correctly); check the first full
  video produced with this version.

## Licences / responsible use

- The hair/face segmentation uses `jonathandinu/face-parsing` (SegFormer, trained on CelebAMask-HQ). That model has **no stated
  licence** and its training data is for **non-commercial research** - check this before any commercial use.
- Only swap faces of people who have given consent. Do not use this for public figures or impersonation.
