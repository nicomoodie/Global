"""Per-frame global camera motion (similarity transform) for a clip, plus jitter stats."""
import sys, json
import cv2, numpy as np

def trajectory(path):
    cap = cv2.VideoCapture(path)
    ok, prev = cap.read()
    prev_g = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    T = []  # dx, dy, da, ds per step
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        pts = cv2.goodFeaturesToTrack(prev_g, 400, 0.01, 20)
        nxt, st, _ = cv2.calcOpticalFlowPyrLK(prev_g, g, pts, None)
        p0, p1 = pts[st == 1], nxt[st == 1]
        M, _ = cv2.estimateAffinePartial2D(p0, p1, method=cv2.RANSAC)
        dx, dy = M[0, 2], M[1, 2]
        da = np.arctan2(M[1, 0], M[0, 0])
        ds = np.hypot(M[0, 0], M[1, 0])
        T.append((dx, dy, da, ds))
        prev_g = g
    return np.array(T)

def smooth(x, r):
    k = np.ones(2 * r + 1) / (2 * r + 1)
    xp = np.pad(x, r, mode='edge')
    return np.convolve(xp, k, mode='valid')

out = {}
for path in sys.argv[1:]:
    T = trajectory(path)
    cum = np.cumsum(T[:, :3], axis=0)
    zoom = np.cumprod(T[:, 3])
    res = {}
    for i, name in enumerate(['x_px', 'y_px', 'roll_rad']):
        c = cum[:, i]
        jitter = c - smooth(c, 12)  # remove motion slower than ~1 s
        res[name] = dict(total=round(float(c[-1]), 2), jitter_rms=round(float(jitter.std()), 3),
                         jitter_peak=round(float(np.abs(jitter).max()), 3))
        if name == 'y_px':
            sp = np.abs(np.fft.rfft(jitter - jitter.mean()))
            f = np.fft.rfftfreq(len(jitter), 1 / 24)
            res[name]['dominant_hz'] = round(float(f[1:][sp[1:].argmax()]), 2)
    res['zoom_total'] = round(float(zoom[-1]), 4)
    # speed profile: forward motion per second (zoom growth per 24 frames)
    lz = np.log(zoom)
    res['zoom_per_sec'] = [round(float(np.exp(lz[min(i + 23, len(lz) - 1)] - (lz[i - 1] if i else 0)) - 1), 4) for i in range(0, len(lz), 24)]
    out[path.split('/')[-1]] = res
print(json.dumps(out, indent=1))
