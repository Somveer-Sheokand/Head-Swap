"""Added head wobble measured with dense optical flow (much less noisy than a face detector).
For every pair of consecutive head frames, fit a similarity transform to the flow inside the face area, once on the
original video and once on the swapped video; the difference is motion the swap added.
usage: python wobble.py track.json original.mp4 swapped1.mp4 [...]"""
import sys, json, cv2, numpy as np
track = json.load(open(sys.argv[1]))["track"]
boxes = [t["box"] for t in track]


def read(p):
    c = cv2.VideoCapture(p)
    out = []
    while True:
        ok, f = c.read()
        if not ok:
            break
        out.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))
    return out


def sim(a, b, box):
    x0, y0, x1, y1 = [int(v) for v in box]
    w, h = x1 - x0, y1 - y0
    X0, Y0 = max(0, x0 - w // 6), max(0, y0 - h // 3)
    X1, Y1 = min(a.shape[1], x1 + w // 6), min(a.shape[0], y1 + h // 10)
    sc = min(1.0, 360.0 / max(X1 - X0, Y1 - Y0))
    ga = cv2.resize(a[Y0:Y1, X0:X1], None, fx=sc, fy=sc)
    gb = cv2.resize(b[Y0:Y1, X0:X1], None, fx=sc, fy=sc)
    fl = cv2.calcOpticalFlowFarneback(ga, gb, None, 0.5, 3, 15, 3, 5, 1.1, 0)
    hh, ww = ga.shape
    ys, xs = np.mgrid[0:hh:3, 0:ww:3]
    cx, cy = ((x0 + x1) / 2 - X0) * sc, ((y0 + y1) / 2 - Y0) * sc
    sel = ((xs - cx) / (0.45 * w * sc)) ** 2 + ((ys - cy) / (0.6 * h * sc)) ** 2 < 1
    src = np.stack([xs[sel], ys[sel]], 1).astype(np.float32)
    A, _ = cv2.estimateAffinePartial2D(src, src + fl[ys[sel], xs[sel]], method=cv2.RANSAC, ransacReprojThreshold=1.0)
    s = float(np.hypot(A[0, 0], A[1, 0]))
    return A[0, 2] / sc, A[1, 2] / sc, np.log(s)


orig = read(sys.argv[2])
O = {}
for i in range(len(orig) - 1):
    if boxes[i] and boxes[i + 1]:
        O[i] = sim(orig[i], orig[i + 1], boxes[i])
for p in sys.argv[3:]:
    v = read(p)
    dt, ds = [], []
    for i, o in O.items():
        if i + 1 < len(v):
            q = sim(v[i], v[i + 1], boxes[i])
            dt.append(np.hypot(q[0] - o[0], q[1] - o[1]))
            ds.append(q[2] - o[2])
    dt, ds = np.array(dt), np.array(ds)
    print(f"{p.split('/')[-1]:30s} added_shift_rms={np.sqrt((dt**2).mean()):.2f}px  p90={np.percentile(dt, 90):.2f}px  "
          f"added_scale_rms={np.sqrt((ds**2).mean())*100:.2f}%  n={len(dt)}")
