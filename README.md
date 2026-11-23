# Qwen-Image-2.1 head swap / face swap API (Modal)

This is a prompt-driven head swap and face swap service built on **Qwen-Image-2.1** (multi-reference image editing), served on Modal GPUs.

You send it a template (a poster) and a face photo. It returns the template at its **original size** with only the head changed. Clothes, body, background, text and logo stay pixel-identical, because only the region the model actually changed (the head) is blended back. That region is found automatically, so nothing is hardcoded per template.

- **Deployed API:** `https://ashish-kumar--qwen-image-2-1-api.us-east.modal.run`
- **Modal app:** `qwen-image-2-1` (Ashish Kumar's workspace)

## API

### Generation
```bash
curl --location 'https://ashish-kumar--qwen-image-2-1-api.us-east.modal.run/jobs' \
--header 'Content-Type: application/json' \
--data '{
  "image_url": "https://YOUR-TEMPLATE-IMAGE.png",
  "source_face_url": "https://YOUR-FACE-PHOTO.jpg",
  "prompt": "optional - see prompts/; the generic face-swap prompt is used when omitted"
}'
```
```json
{"job_id":"fc-...","status":"pending","status_url":"https://.../jobs/fc-..."}
```

Optional fields:

| Field | Default | Meaning |
|---|---|---|
| `seed` | random | Fixes the variation. Run 2 seeds and keep the better result. |
| `model` | `base` | `base` gives the best quality (40 steps). `turbo` is faster (8 steps). |
| `size` | `auto` | `auto` is the largest size that fits in GPU memory: 1600px for base, 2048px for turbo. |
| `preserve_original` | `true` | Return the template at its original size with only the head changed. |

### Status
```bash
curl --location 'https://ashish-kumar--qwen-image-2-1-api.us-east.modal.run/jobs/JOB_ID'
```
```json
{"job_id":"fc-...","status":"completed","download_url":"https://.../v1/headswap/fc-.../image"}
```
`status` is `pending`, `completed` or `failed`. A failed job includes an `error` field.

### Other endpoints
| Endpoint | Purpose |
|---|---|
| `GET /v1/headswap/prompt` | The current default (generic) prompt. |
| `POST /upload` (multipart `file`) → `{"url"}` | Host an image to use as `image_url` / `source_face_url`. Upload large posters as JPEG. |
| `GET /files/{key}` | Serves an uploaded image. |
| `POST /v1/headswap` (multipart) | The same pipeline with file uploads instead of URLs. |
| `POST /v1/images/generations`, `POST /v1/images/edits` | OpenAI-Images-compatible text-to-image and edit endpoints. |
| `GET /health` | Health check. |

## Prompts (`prompts/`)

| File | Use |
|---|---|
| `face_swap_generic.txt` | **Default.** Works on any template: copies the pose, lighting, hair, ears and clothes from the template, and the identity and facial hair from the face photo. |
| `male_template_pose_lighting.txt` | Example of a template-specific prompt. For dramatic poses (raised chin) and strong side light, describing the pose and light explicitly gives clearly better results. |
| `face_swap_keep_hair_poster.txt` | Face swap that keeps the template's hairstyle and glasses (peanut-butter poster example). |

Lessons from testing:
- **Words become objects.** Writing "earrings" adds earrings to men, and "full beard" adds a beard to clean-shaven faces. Use neutral wording such as "everything worn on the ears" and "copy the facial hair exactly; add none".
- **Put identity first.** Too many "keep everything the same" lines make the model keep the original person.
- **Expect some failures.** About 1 in 4–8 runs returns the original face. Send 2 seeds and keep the better result.

## Files

| File | What it does |
|---|---|
| `qwen_image_modal.py` | Modal app: weight download, the `QwenImage` GPU class (turbo/base), the `headswap_job` CPU orchestrator, the `preserve_original` blend and the FastAPI app. |
| `run_jobs.py` | Calls the API exactly like the curl above (`/jobs` → `status_url` → `download_url`), with network retries and request/response logs. |
| `poster_headswap.py` | Head swaps one poster with several faces, models and seeds. |
| `keep_hair.py` | Optional post-step: keeps the template's original hair pixel-identical and takes only the face skin from the swap. |

## Deploy
```bash
modal run qwen_image_modal.py::download --variant turbo   # weights -> Volume "qwen-image-hf-cache" (CPU only)
modal run qwen_image_modal.py::download --variant base
modal deploy qwen_image_modal.py
```
The weights are cached in the Modal Volume `qwen-image-hf-cache`, and uploaded images in the Modal Dict `qwen-headswap-uploads`. GPU spend is capped at `max_containers=4` H100s. The API is public, with no auth.

## Performance and cost (H100)
- **Base, 1600px, 2 reference images:** about 50–100s end to end, about $0.03 per image.
- **Turbo, 2048px:** about 20–30s end to end, about $0.015 per image.
- **Base and 2048px:** base with 2 reference images runs out of GPU memory above 1600px, so `auto` caps it.

**License:** the Qwen-Image-2.1 weights use the Qwen Research License. Check its terms before commercial use.
