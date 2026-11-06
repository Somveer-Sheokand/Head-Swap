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
`/swap` options: `start_frame`, `max_frames` (process only part of the video), `max_cost_usd`, `output_put_url`,
`quality` (`"best"` = every head frame at 1024 px; anything else = cheaper, every 2nd frame at 832 px + interpolation).
`POST /swap-video` does preprocess + swap in one call (`video_url`, `target_face_url`, `source_face_url`).

## Measured time and cost (L40S, ~US$1.95/h)

| Job | Time | Approx. cost |
|---|---|---|
| Image (1080x1920) | ~1-2 min (mostly cold start) | ~$0.03-0.07 |
| Video, 15.5 s, `best` | ~30 min (15 min GPU + 15 min CPU blending) | ~$1.5-2 |

Very large images (tens of megapixels) are slower because blending runs on the full image.

## Known limitations

- Each frame's head is generated independently, so some frame-to-frame shimmer remains.
- In some side-profile frames a thin sliver of the old hair can remain at the very edge of the picture.
- Fast zooms warp in the cheaper (`quality` != `best`) mode.

## Licences / responsible use

- The hair/face segmentation uses `jonathandinu/face-parsing` (SegFormer, trained on CelebAMask-HQ). That model has **no stated
  licence** and its training data is for **non-commercial research** - check this before any commercial use.
- Only swap faces of people who have given consent. Do not use this for public figures or impersonation.
