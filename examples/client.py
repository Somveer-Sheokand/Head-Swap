"""Example client for the Klein head swap HTTP API.

    python examples/client.py --api https://<workspace>--headswap-klein-api.modal.run --token <API_TOKEN> \
        --body poster.png --head face.jpg --out result.png
"""
import argparse

import requests


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", required=True)
    ap.add_argument("--token", required=True, help="API_TOKEN from the klein-api-token Modal secret")
    ap.add_argument("--body", required=True, help="poster / target image file")
    ap.add_argument("--head", required=True, help="face image file")
    ap.add_argument("--out", default="result.png")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mode", default="crop", choices=["crop", "full"])
    ap.add_argument("--extra-prompt", default="")
    a = ap.parse_args()

    with open(a.body, "rb") as b, open(a.head, "rb") as h:
        r = requests.post(f"{a.api}/headswap", headers={"Authorization": f"Bearer {a.token}"},
                          files={"body": b, "head": h},
                          data={"mode": a.mode, "seed": a.seed, "extra_prompt": a.extra_prompt}, timeout=900)
    r.raise_for_status()
    open(a.out, "wb").write(r.content)
    print("saved", a.out)


if __name__ == "__main__":
    main()
