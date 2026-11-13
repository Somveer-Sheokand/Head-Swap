# Video head swap with LTX-2.5 (BFS-Best-Face-Swap-Video)

Branch `ltx25-video-headswap`. Separate from the production BFS image pipeline (`bfs_service.py` /
`bfs-headswap-api`, unchanged).

## What it is
[BFS-Best-Face-Swap-Video](https://huggingface.co/Alissonerdx/BFS-Best-Face-Swap-Video) (LoRA `head_swap_ltx25_r128_v1.1`)
on [Lightricks LTX-2.5](https://huggingface.co/Lightricks/LTX-2.5) (22B distilled, int8), run through the author's ComfyUI
workflow (`workflows_bfs_video_ltx25_v1.1.json`, converted to API format at run time). The head is generated as **one
video per shot**, so there is no per-frame jitter.

Licence: LTX-2.x Community License (free commercial use under $10M annual revenue). The model repo is gated: accept
the licence on Hugging Face and pass a token to `download_ltx` once.

## Files
| File | What it is |
|---|---|
| `research_service.py` | Modal app `headswap-research`: `LTXHeadSwapWorker` (GPU, ComfyUI + LTX-2.5), `compose_head` (CPU paste-back), `download_ltx`, placeholder endpoints |
| `ltx_headswap.py` | local end-to-end script: shots, crops, 1:1 guides, generation, paste-back, audio |
| `workflows_bfs_video_ltx25_v1.1.json` | the model author's ComfyUI workflow |

## Setup
```bash
modal deploy research_service.py
python -c "import modal; print(modal.Function.from_name('headswap-research','download_ltx').remote(hf_token='<token>'))"
```

## Run
```bash
python ltx_headswap.py --video video.mp4 --track track.json --ref head.png \
  --face "Male, medium-brown skin, 40-50 years old, dark brown eyes, very short buzz-cut hair, light stubble." \
  --out result.mp4 --dry-run          # prints shots, crops and the cost estimate; drop --dry-run to generate
```
`video.mp4` / `track.json` are the files `/preprocess` of the BFS API writes per template.

## Pipeline and why
| Step | Fixes |
|---|---|
| Split into camera shots (colour-histogram change or one-frame spike) | identity jumps across cuts |
| One fixed 9:16 crop per shot, small faces zoomed in | wide shots not swapped (face too small) |
| Original frames written 1:1 at 24 fps, padded by holding the last frame | repeated / skipped frames from 24<->25 fps conversion |
| Shots run one after another in one GPU container | models load once (cost) |
| Head + hair face-parsing mask, colour matched (+-3 frames) | seams on hands / food, text and logo changes, skin-tone mismatch |

## Results (Burger King test, 15.5 s, 4 shots)
| | Original | BFS pipeline | LTX-2.5 v2 |
|---|---|---|---|
| Head flicker (lower = steadier) | 7.11 | 6.70 | **5.93** |
| Swap in all shots incl. wide shot | - | yes | yes |
| Expressions follow the original | - | yes | mostly: the LoRA invents some expressions (eyes open while biting, smiling at the end of the wide shot) |

Cost: about $0.15-0.25 per 2-3 s shot plus one model load on an L40S (~$0.5-0.6 for this video); `--dry-run` prints the estimate.

## Known issues / next steps
- Expressions: add a per-shot ACTION description to the prompt (author's V3 prompt format).
- Group shots without a close crop are softer than the BFS pipeline.
