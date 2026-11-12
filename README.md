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
`/swap` options: `start_frame`, `max_frames` (process only part of the video), `max_cost_usd` (default 5),
`output_put_url`, `dry_run` (only returns frame count + cost estimate, no GPU), and `quality`:

| `quality` | What it does |
|---|---|
| `"best"` (**default**) | 1024 px, every head frame generated - sharpest result, needed for close-up / large heads. Plus a second motion-aligned smoothing pass over +-3 frames (`post_taps`) and a light temporal denoise at the join |
| `"fast"` | up to 832 px (smaller for small / far-away heads, min 640 px), every 2nd frame generated, frames in between carried by motion (frames where that fails are generated too). About 2x cheaper, but visibly softer on close-ups |

`POST /swap-video` does preprocess + swap in one call (`video_url`, `target_face_url`, `source_face_url`).
`/swap-image` uses the same `quality` values (default `"best"`).

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
| Image (1080x1920), `best` | ~2-2.5 min (mostly cold start) | ~$0.12-0.20 |
| Video, 8 s (193 frames), `best` | ~18-22 min | ~$3.5-4.3 |
| Video, 8 s (193 frames), `fast` | ~12 min (measured) | ~$1.3-1.5 (measured) |

An 8 s `best` video fits under the default `max_cost_usd` of 5; longer videos need a higher limit or
`start_frame`/`max_frames`.

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

## Jitter / wobble tests (15.5 s Burger King ad, 208 head frames, same generated heads)

Head flicker = mean |frame - average of its two neighbours| in the head area (lower = steadier). Original footage: **7.10**.

| Variant | Head flicker |
|---|---|
| previous `best` (2-frame blend) | 7.98 |
| + position/scale wobble correction (`align_fix`) | 8.95 - worse, left off |
| + second smoothing pass +-2 frames | 7.54 |
| + stronger second pass (`post_taps=3`, or +-2 with heavier weights) | 7.36-7.39 |
| **+ temporal denoise (now the `best` default)** | **6.87-6.89** |

**Wobble (head stabiliser + colour smoothing, now in `best`):** measured with dense optical flow inside the face
(frame-to-frame motion of the swapped face minus the original face's motion; `wobble.py`-style check):

| Variant | Added shift | Added scale | Head flicker |
|---|---|---|---|
| previous `best` | 8.47 px | 0.80 % | 6.87 |
| + colour smoothing (`cm_smooth=2`) | 8.17 px | 0.79 % | 6.86 |
| + head stabiliser (`stab_iters=2`) | 7.73 px | 0.68 % | 6.72 |
| **+ both (now the default)** | **7.25 px** | **0.64 %** | **6.70** |
| stronger (3 passes, +-3 frames, full strength) | 7.28 px | 0.65 % | 6.66 - no further gain |

The stabiliser registers each generated head (similarity transform fitted to dense flow inside the head mask) onto
the motion-compensated average of its neighbours, damped and clamped (max 3 % shift, 4 % scale, 3 degrees per frame).
It adds ~3-5 min of CPU blending per video (parallel slices, `BLEND_PAD` raised to 10 frames).
The face-detector based correction (`align_fix`) tested worse and stays off.

No ghosting was visible in close-up eating shots. The second pass rejects neighbour pixels that do not match
(fast motion, hands/food in front of the face), so it does not smear.

### Experimental: video-model refine (`refine_video`, not used by default)

`refine_video(template_id, video_name, denoise=0.25, crop_res=1024, crop_scale=0.72)` re-draws the swapped head of a
finished video with **Wan 2.2 TI2V-5B** (Apache-2.0, video-to-video at low strength, up to 81 frames at once) and pastes
only the head/hair back. Test on an 81-frame close-up (L40S, ~$0.25 per clip):

| Version | Head flicker | Sharpness |
|---|---|---|
| current `best` | 6.53 | 12.2 |
| Wan refine 0.25 | 5.92 | 8.8 (softer) |
| Wan refine + `best` fine detail added back | 5.94 | 12.8 |

Flicker drops ~9 %, but the video model changes identity details (eye colour drifted from blue-grey to brown,
eyebrow shape) and smooths skin, so it is **not** enabled. Models are in `download_models.py`.

### Optional: keyframe propagation (`prop=True, flow="raft"`, not default)

BFS heads only on an absolute keyframe grid (`key_step`) plus frames where warping the original key frame with RAFT
optical flow (torchvision, BSD) does not reproduce the footage; frames in between carry the key heads + masks along
the original motion. Same ad, same heads:

| Version | Head flicker | Added shift (p90) / scale | Notes |
|---|---|---|---|
| `best` (v4) | 6.70 | 7.25 px (12.3) / 0.64 % | default |
| keys every 4th frame | 6.99 | 8.08 px (13.5) / 0.78 % | warping errors in a fast side-profile turn |
| keys every 8th frame | 6.85 | **6.70 px (9.9) / 0.44 %** | least wobble, but faint seams on the forehead in fast motion |

With `prop` only the key frames need the GPU, so it is also a way to cut GPU cost (~50-85 % fewer generated frames).

### Experimental: video-native head replacement (`animate_video`, Wan 2.2 Animate 14B, not used by default)

Head generated as one video (up to 77-81 frames) in head-crop space: reference = most frontal BFS frame,
face_video = face crops (expressions), pose_video = DWPose (comfyui_controlnet_aux), background = crop with the head
blacked out + head mask. Apache-2.0 model; ~7.5 min L40S per 81 frames (~$0.25). Close-up test:

| Version | Head flicker | Added shift / scale (dense flow) | Sharpness | Likeness |
|---|---|---|---|---|
| `best` (BFS + smoothing + stabiliser) | 6.24 | 11.45 px / 0.63 % | 11.4 | actor |
| Animate, face crops from original | 6.78 | 5.80 px / 0.48 % | 16.8 | **original man** |
| Animate, face crops from BFS | 6.80 | 6.87 px / 0.45 % | - | **still mostly the original man**, stronger expressions |

Much less wobble and sharper, but the identity of the reference is not kept, so it is not enabled.

### Experimental: HyperSwap face swap (`hyperswap_video`, not used by default)

FaceFusion `hyperswap_1a_256` (ONNX, CPU, **licence ResearchRAIL - research use**), with smoothed 5-point landmarks,
768 px pixel boost and a face-parsing mask (keeps hands / food). Same 81-frame close-up:

| Version | Head flicker | Sharpness | Problems |
|---|---|---|---|
| current `best` (BFS) | 6.53 | 12.2 | - |
| HyperSwap only | 5.81 | 15.3 | keeps the original man's hair / head shape; **opens the eyes when the actor closes them** |
| BFS + HyperSwap | 6.43 | 10.8 | same eye problem, softer |

Steadier and cheaper (~$0.05 per clip), but expressions and likeness are worse, so BFS stays the default.

## Known limitations

- Each frame's head is generated independently; in `best` the smoothing passes bring the head flicker to about the
  level of the original footage, but very fine hair detail can still shimmer slightly.
- In some side-profile frames a thin sliver of the old hair can remain at the very edge of the picture.
- Very fast zooms / head turns can warp slightly in `fast` mode; use `"quality":"best"` for such footage.

## Licences / responsible use

- The hair/face segmentation uses `jonathandinu/face-parsing` (SegFormer, trained on CelebAMask-HQ). That model has **no stated
  licence** and its training data is for **non-commercial research** - check this before any commercial use.
- Only swap faces of people who have given consent. Do not use this for public figures or impersonation.
