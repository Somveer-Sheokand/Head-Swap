# Image Head Swap Studio

Best-quality **image head swap** for ad creatives, built on open-source models and hosted on [Modal](https://modal.com).
Only the head (and neck) of the person in a poster is replaced; clothes, jewellery, text, logo and background stay
**pixel-exact**.

> Use only faces you have the rights to use (signed release / licensed stock). Do not use this for impersonation.

## How it works

```
poster + face photo
   │
   ├─ 1. head crop around the target person (InsightFace)
   ├─ 2. generate N candidates          engine: Klein (Flux 2 Klein 4B + BFS head)  [default]
   │                                    or     Qwen Image 2.1 + BFS head v1.1      ("bfs")
   │                                    or     Qwen-Image-Edit-2511 + BFS head V5  ("edit", prompt-controlled)
   ├─ 3. pick the best candidate        identity (ArcFace), expression + head pose (68 landmarks), mouth, extra hair
   ├─ 4. paste head+neck onto the ORIGINAL full-resolution image
   │      - clothes / collars / necklaces protected (face parsing + colour check below the chin)
   │      - old hair / ponytail removed, background around the head matched
   │      - skin tone + light matched, soft neck transition
   ├─ 5. finishing: face skin -> original model's face tone, neck -> original neck tone (two zones)
   │      everything outside the head/neck restored bit-exact
   └─ 6. quality gate: clothes_px == 0, |face_tone| <= 6, jaw->neck misjoin <= 10
          -> neck-profile repair, else regenerate with new seeds (max_rounds)
```

| Engine | Strength | Weakness |
|---|---|---|
| `klein` (default) | most photographic skin/hair, keeps the head angle, ~15 s, ~$0.01 / candidate | weaker likeness on some faces |
| `bfs` | strongest likeness | slightly smoother "AI" look |
| `edit` | prompt-controllable (lips, expression, glasses) | can look over-sharp / plastic |

## Files

| File | What |
|---|---|
| `headswap_studio.py` | Modal app `headswap-studio`: GPU workers, finishing pipeline, quality gate, HTTP API |
| `klein_headswap.py` | Stand-alone Flux 2 Klein 4B + BFS head swap app (simple crop + paste) |
| `download_models.py` | Downloads Qwen Image 2.1 + BFS head v1.1 into the `bfs-models` volume |
| `download_edit_models.py` | Downloads Qwen-Image-Edit-2511 + BFS head V5 + Lightning into the same volume |
| `examples/client.py` | Example client: submit a job, wait, download the result |

## Setup

```bash
pip install modal
modal token set --token-id <id> --token-secret <secret>      # your own Modal token, never commit it
modal run download_models.py::download                      # ~18 GB, once
modal run download_edit_models.py::download                 # ~31 GB, once (only for engine "edit")
modal deploy headswap_studio.py
```

If your Modal workspace uses a newer image builder than your client supports, deploy with
`MODAL_IMAGE_BUILDER_VERSION=2025.06`.

## API

`POST https://<workspace>--headswap-studio-api.us-east.modal.run/headswap`

```json
{
  "image_url": "https://.../poster.png",
  "face_url": "https://.../face.jpg",
  "best_of": 4,
  "neck_depth": 1.0,
  "engine": "klein",
  "max_rounds": 2
}
```

| Field | Default | Meaning |
|---|---|---|
| `image_url`, `face_url` | required | poster / new face (public URLs) |
| `engine` | `klein` | `klein`, `bfs` or `edit` |
| `best_of` | 4 | candidates per round (1-6) |
| `max_rounds` | 2 | extra rounds with new seeds if the quality gate fails (1-3) |
| `neck_depth` | 1.0 | how far down the new neck goes: `0.76` if a necklace sits on the neck, `1.0-1.15` otherwise |
| `target_face_url` / `face_index` | – | which person to swap when there are several |
| `prompt_extra` | – | extra instructions (engine `edit`), e.g. `"nude lips, closed mouth"` |

Then poll `GET /jobs/{job_id}` until `status == "done"` and download `result.download_url`.
The result contains `passed_quality_gate`, the measured `quality` numbers and every tried candidate.

## Tips for the best results

- Face photo: high resolution (1000 px+), neutral expression, soft light, **no glasses / heavy lipstick / big earrings**
  (the models copy them).
- Expression and head angle are copied from the poster; a very different face photo angle lowers likeness.
- For hero print ads, review at 100 % and give the jawline / hairline a short retouch if needed.

## Cost (Modal L40S)

`klein`: ~$0.01 per candidate · `bfs`: ~$0.02-0.03 per candidate · cold start 1-3 min after idle.
