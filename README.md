# Image Head Swap – Flux 2 Klein 4B + BFS

Image head swap on **FLUX.2-klein-4B** (Apache-2.0, 4-step distilled) with the **BFS head LoRA**
([Alissonerdx/BFS-Best-Face-Swap](https://huggingface.co/Alissonerdx/BFS-Best-Face-Swap)), hosted on [Modal](https://modal.com).

> Use only faces you have the rights to use (signed release / licensed stock). Do not use this for impersonation.

## Why Klein

In our tests on ad posters it gave the most photographic result of the open-source engines: natural skin texture and
hair, and it keeps the original head angle (e.g. a chin-up pose). It is also fast (~12-15 s per image on a warm L40S,
~$0.01 per image).

## How it works

- **Picture 1 = body / target poster, Picture 2 = head / face to insert** (never reverse them).
- Prompt: the official BFS Klein prompt (`head_swap: start with Picture 1 as the base image ...`) + optional extra text.
- `mode=crop` (default): the face is found with OpenCV YuNet, a square head crop is swapped, and only the changed
  head/neck area is pasted back onto the **full-resolution original** (text, logos and detail stay sharp).
- `mode=full`: the whole image is regenerated at `max_side` (like the plain ComfyUI workflow).

## Deploy

```bash
pip install modal
modal token set --token-id <id> --token-secret <secret>          # your own Modal token, never commit it
modal secret create klein-api-token API_TOKEN=<choose-a-token>   # protects the HTTP API
modal deploy klein_headswap.py
```

## Use

Command line (no HTTP):

```bash
modal run klein_headswap.py --body poster.png --head face.jpg --out result.png
```

HTTP API (Bearer token = the `API_TOKEN` above):

```bash
curl -X POST https://<workspace>--headswap-klein-api.modal.run/headswap \
  -H "Authorization: Bearer <API_TOKEN>" \
  -F body=@poster.png -F head=@face.jpg \
  -F mode=crop -F lora=v1 -F lora_strength=1.0 -F steps=4 -F seed=42 \
  -F extra_prompt="calm closed-mouth expression" \
  -o result.png
```

`GET /` serves a small upload page.

| Field | Default | Meaning |
|---|---|---|
| `body`, `head` | required | poster image / face image (files) |
| `mode` | `crop` | `crop` = paste head onto the full-res original, `full` = regenerate the whole image |
| `lora` | `v1` | `v1` or `v1.1` BFS Klein LoRA |
| `lora_strength` | 1.0 | 0-2 |
| `steps` | 4 | 1-50 (model is distilled for 4) |
| `seed` | 42 | change for a different variation |
| `extra_prompt` | – | e.g. `"nude lips, closed mouth, no glasses"` |
| `max_side` | 1024 | generation size (256-2048) |

See `examples/client.py` for a Python client.

## Tips

- Face photo: high resolution, neutral expression, soft light, no glasses / heavy lipstick / big earrings (copied).
- Try 2-4 seeds and keep the best one.
- Klein does not match skin tone to the body or protect collars on its own; check the neck and collar at 100 %.
