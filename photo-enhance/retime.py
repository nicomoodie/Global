"""Re-time a clip to constant on-screen motion, measured directly from the picture.

For moves the dolly model in rigify.py does not describe (orbits, arcs, sweeping turns):
per-frame speed = median optical-flow length of tracked features, then output frames are
sampled at equal steps of accumulated motion (blending neighbouring source frames).
--metric dense uses the mean dense (Farneback) flow over the whole frame, which follows perceived
motion in orbits with strong parallax better than the median of sparse features.
--metric travel is for forward dollies: per-frame expansion rate (median radial flow / radius
about the frame centre) is integrated to log-scale L, and frames are sampled at equal steps of
camera travel 1 - exp(-L), i.e. constant dolly speed.
No stabilisation warp, so nothing in the picture is bent; output 1920x1080.

Usage: python3 -I retime.py IN.mp4 OUT.mp4 --seconds 7 [--trim-start 0] [--trim-end 0] [--ease 0] [--metric travel]
"""
import argparse
import subprocess

import cv2
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('src')
ap.add_argument('dst')
ap.add_argument('--seconds', type=float, required=True)
ap.add_argument('--fps', type=int, default=24)
ap.add_argument('--trim-start', type=int, default=0, help='source frames dropped at the start')
ap.add_argument('--trim-end', type=int, default=0, help='source frames dropped at the end')
ap.add_argument('--ease', type=float, default=0.0, help='share of ease-in/out at each end; 0 = constant speed')
ap.add_argument('--smooth', type=float, default=4.0, help='gaussian sigma (frames) applied to the speed curve')
ap.add_argument('--metric', choices=['flow', 'dense', 'travel'], default='flow',
                help='flow/dense: constant on-screen motion (orbits, pans); travel: constant forward dolly speed')
a = ap.parse_args()

cap = cv2.VideoCapture(a.src)
frames = []
while True:
    ok, f = cap.read()
    if not ok:
        break
    frames.append(f)
frames = frames[a.trim_start:len(frames) - a.trim_end if a.trim_end else None]
h, w = frames[0].shape[:2]

speed = []
prev = cv2.cvtColor(cv2.resize(frames[0], (960, 540)), cv2.COLOR_BGR2GRAY)
centre = np.array([480.0, 270.0])
for f in frames[1:]:
    g = cv2.cvtColor(cv2.resize(f, (960, 540)), cv2.COLOR_BGR2GRAY)
    if a.metric == 'flow':
        p0 = cv2.goodFeaturesToTrack(prev, 600, 0.005, 8)
        p1, st, _ = cv2.calcOpticalFlowPyrLK(prev, g, p0, None)
        ok = st.ravel() == 1
        speed.append(float(np.median(np.linalg.norm((p1[ok] - p0[ok]).reshape(-1, 2), axis=1))) if ok.sum() > 20 else 0.0)
    elif a.metric == 'dense':
        flow = cv2.calcOpticalFlowFarneback(prev, g, None, 0.5, 4, 21, 3, 5, 1.1, 0)
        speed.append(float(np.linalg.norm(flow, axis=2).mean()))
    else:
        p0 = cv2.goodFeaturesToTrack(prev, 1000, 0.005, 6)
        p1, st, _ = cv2.calcOpticalFlowPyrLK(prev, g, p0, None, winSize=(21, 21), maxLevel=4)
        pb, st2, _ = cv2.calcOpticalFlowPyrLK(g, prev, p1, None, winSize=(21, 21), maxLevel=4)
        ok = (st.ravel() == 1) & (st2.ravel() == 1) & (np.linalg.norm((pb - p0).reshape(-1, 2), axis=1) < 0.3)
        r = p0[ok].reshape(-1, 2) - centre
        d = p1[ok].reshape(-1, 2) - p0[ok].reshape(-1, 2)
        rn = np.linalg.norm(r, axis=1)
        m = rn > 60
        speed.append(float(np.median((d[m] * r[m]).sum(1) / rn[m] ** 2)) if m.sum() > 20 else 0.0)
    prev = g
speed = np.array(speed)
if a.metric == 'travel':
    log_s = np.concatenate([[0], np.cumsum(np.maximum(speed, 0))])
    speed = np.diff(1 - np.exp(-log_s)) * 1000
if a.smooth > 0:
    r = int(3 * a.smooth)
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / a.smooth) ** 2)
    k /= k.sum()
    speed = np.convolve(np.pad(speed, r, mode='edge'), k, mode='valid')
prog = np.concatenate([[0], np.cumsum(speed + 1e-6)])
prog /= prog[-1]

n_out = int(round(a.seconds * a.fps))
u = np.linspace(0, 1, n_out)
e = a.ease
if e > 0:
    tgt = np.interp(u, [0, e, 1 - e, 1], [0, e / 2, 1 - e / 2, 1])
    tgt = np.where(u < e, (e / 2) * (u / e) ** 2, tgt)
    tgt = np.where(u > 1 - e, 1 - (e / 2) * ((1 - u) / e) ** 2, tgt)
else:
    tgt = u
t_src = np.interp(tgt, prog, np.arange(len(prog)))

# centre-crop to exact 16:9, then scale to 1920x1080
cw = min(w, int(round(h * 16 / 9)))
ch = int(round(cw * 9 / 16))
x0, y0 = (w - cw) // 2, (h - ch) // 2
out_w, out_h = 1920, 1080
enc = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24',
                        '-s', f'{out_w}x{out_h}', '-r', str(a.fps), '-i', '-',
                        '-c:v', 'libx264', '-crf', '15', '-preset', 'slow', '-pix_fmt', 'yuv420p',
                        '-movflags', '+faststart', a.dst], stdin=subprocess.PIPE)
last = len(frames) - 1
for t in t_src:
    i0 = int(np.floor(t))
    i1 = min(i0 + 1, last)
    fr = t - i0
    img = (1 - fr) * frames[i0].astype(np.float32) + fr * frames[i1].astype(np.float32)
    img = cv2.resize(img[y0:y0 + ch, x0:x0 + cw], (out_w, out_h), interpolation=cv2.INTER_LANCZOS4)
    enc.stdin.write(np.clip(img + 0.5, 0, 255).astype(np.uint8).tobytes())
enc.stdin.close()
enc.wait()
print(f'{a.src}: {len(frames)} src frames -> {n_out} out frames ({len(frames) / a.fps:.1f}s -> {a.seconds:.1f}s), '
      f'speed min/med/max {speed.min():.2f}/{np.median(speed):.2f}/{speed.max():.2f} '
      f'{"travel x1000/frame" if a.metric == "travel" else "px/frame"}')
