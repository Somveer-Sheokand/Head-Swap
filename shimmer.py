"""Head-area flicker: mean |f_t - (f_t-1 + f_t+1)/2| inside the head box (lower = steadier).
usage: python shimmer.py track.json video1.mp4 [video2.mp4 ...]"""
import sys, json, cv2, numpy as np
track = json.load(open(sys.argv[1]))["track"]
boxes = [t["box"] for t in track]
for path in sys.argv[2:]:
    cap = cv2.VideoCapture(path)
    fr = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        fr.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32))
    vals = []
    for i in range(1, len(fr) - 1):
        if not (boxes[i] and boxes[i - 1] and boxes[i + 1]):
            continue
        x0, y0, x1, y1 = [int(v) for v in boxes[i]]
        w, h = x1 - x0, y1 - y0
        x0, x1 = max(0, x0 - w // 4), min(fr[i].shape[1], x1 + w // 4)
        y0, y1 = max(0, y0 - h // 2), min(fr[i].shape[0], y1 + h // 8)
        d = np.abs(fr[i][y0:y1, x0:x1] - 0.5 * (fr[i - 1][y0:y1, x0:x1] + fr[i + 1][y0:y1, x0:x1]))
        vals.append(float(d.mean()))
    print(f"{path.split('/')[-1]:45s} head_flicker={np.mean(vals):.3f}  p90={np.percentile(vals, 90):.3f}  n={len(vals)}")
