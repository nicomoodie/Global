"""Make an AI 'walk-through' clip feel like a motorized dolly shot.

1. Track global camera motion (similarity transform per frame).
2. Smooth the camera path heavily -> removes footstep bob / sway, keeps the move.
3. Warp each frame onto the smoothed path, then crop-zoom just enough to hide borders.
4. Re-time so apparent motion runs at constant speed (soft ease at the ends),
   and speed the clip up to the requested output length.

Usage: python3 -I rigify.py IN.mp4 OUT.mp4 --seconds 5 [--sigma 18] [--path fep|smooth|rail] [--ease 0]
"""
import argparse
import subprocess

import cv2
import numpy as np


def step_transforms(path):
    cap = cv2.VideoCapture(path)
    ok, prev = cap.read()
    h, w = prev.shape[:2]
    pg = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    mats = [np.eye(3)]
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        pts = cv2.goodFeaturesToTrack(pg, 600, 0.01, 15)
        nxt, st, _ = cv2.calcOpticalFlowPyrLK(pg, g, pts, None)
        M, _ = cv2.estimateAffinePartial2D(pts[st == 1], nxt[st == 1], method=cv2.RANSAC,
                                           ransacReprojThreshold=2.0)
        mats.append(np.vstack([M, [0, 0, 1]]))
        pg = g
    return mats, w, h


def params(M):
    s = np.hypot(M[0, 0], M[1, 0])
    a = np.arctan2(M[1, 0], M[0, 0])
    return np.array([M[0, 2], M[1, 2], a, np.log(s)])


def matrix(p):
    x, y, a, ls = p
    s = np.exp(ls)
    return np.array([[s * np.cos(a), -s * np.sin(a), x], [s * np.sin(a), s * np.cos(a), y], [0, 0, 1]])


def gauss_smooth(x, sigma):
    r = int(3 * sigma)
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    # reflect-pad so the path's ends keep their direction instead of flattening
    xp = np.concatenate([2 * x[0] - x[r:0:-1], x, 2 * x[-1] - x[-2:-r - 2:-1]])
    return np.convolve(xp, k, mode='valid')


def cover_zoom(W, w, h, z_max=1.2):
    """Smallest centred zoom so the output frame stays inside the warped source."""
    cx, cy = w / 2, h / 2
    lo, hi = 1.0, z_max
    for _ in range(25):
        z = (lo + hi) / 2
        corners = np.array([[cx + (u - cx) / z, cy + (v - cy) / z, 1]
                            for u, v in [(0, 0), (w, 0), (w, h), (0, h)]]).T
        ok = True
        for Wi in W:
            src = np.linalg.inv(Wi) @ corners
            src = src[:2] / src[2]
            if (src[0] < 0).any() or (src[0] > w).any() or (src[1] < 0).any() or (src[1] > h).any():
                ok = False
                break
        lo, hi = (lo, z) if ok else (z, hi)
    return hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('src')
    ap.add_argument('dst')
    ap.add_argument('--seconds', type=float, default=5.0)
    ap.add_argument('--fps', type=int, default=24)
    ap.add_argument('--sigma', type=float, default=18.0, help='path smoothing, frames')
    ap.add_argument('--ease', type=float, default=0.12, help='share of ease-in/out at each end; 0 = constant speed (cut on motion)')
    ap.add_argument('--path', choices=['smooth', 'rail', 'fep'], default='fep',
                    help="fep: one straight rail = zoom about a single fixed expansion point, roll "
                         "locked, smooth scale (removes bob, sway, weave, banking); "
                         "rail/smooth: older per-parameter fits")
    ap.add_argument('--degree', type=int, default=2, help='rail curve degree')
    ap.add_argument('--no-retime', action='store_true', help='keep original timing (frame i -> frame i)')
    ap.add_argument('--lock-roll', action='store_true', help='smooth path: hold roll at the first frame (level horizon)')
    a = ap.parse_args()

    steps, w, h = step_transforms(a.src)
    C = [np.eye(3)]
    for M in steps[1:]:
        C.append(M @ C[-1])
    P = np.array([params(c) for c in C])
    if a.path == 'fep':
        # A camera on one straight rail (any direction, fixed orientation) moves image content
        # as a pure zoom about ONE fixed point F (the focus of expansion). Fit F to the whole
        # clip, smooth the scale, and warp every frame onto that ideal move.
        Ca = np.array(C)
        c = np.array([w / 2, h / 2, 1.0])
        s_raw = np.hypot(Ca[:, 0, 0], Ca[:, 1, 0])
        u = (Ca @ c)[:, :2] - c[:2]
        e = (s_raw - 1)[:, None]
        g = (e * u).sum(0) / max(float((e * e).sum()), 1e-12)
        F = c[:2] - g
        ls = gauss_smooth(np.log(s_raw), a.sigma)
        ls = np.maximum.accumulate(ls) if ls[-1] >= ls[0] else np.minimum.accumulate(ls)
        sm = np.exp(ls)
        T = [np.array([[k, 0, F[0] * (1 - k)], [0, k, F[1] * (1 - k)], [0, 0, 1]]) for k in sm]
        W = [T[i] @ np.linalg.inv(C[i]) for i in range(len(C))]
        z = cover_zoom(W, w, h)
        speed = np.abs(np.diff(ls)) + 1e-9
        print(f'fep: expansion point F = ({F[0] / w:.3f}W, {F[1] / h:.3f}H), total zoom {sm[-1]:.3f}')
    elif a.path == 'rail':
        t = np.arange(len(P))
        S = np.column_stack([np.polyval(np.polyfit(t, P[:, j], a.degree), t) for j in range(4)])
        S[:, 2] = 0.0  # lock roll: the horizon never tilts
    else:
        S = np.column_stack([gauss_smooth(P[:, j], a.sigma) for j in range(4)])
        if a.lock_roll:
            S[:, 2] = 0.0
    if a.path != 'fep':
        W = [matrix(S[i]) @ np.linalg.inv(C[i]) for i in range(len(C))]
        z = cover_zoom(W, w, h)
        # apparent speed along the smoothed path: translation + zoom seen at the frame corner
        half_diag = np.hypot(w, h) / 2
        d = np.diff(S, axis=0)
        speed = np.hypot(d[:, 0], d[:, 1]) + half_diag * np.abs(d[:, 3]) + half_diag * np.abs(d[:, 2])
    prog = np.concatenate([[0], np.cumsum(speed)])
    prog /= prog[-1]

    n_out = int(round(a.seconds * a.fps))
    u = np.linspace(0, 1, n_out)
    # constant speed with a short smoothstep ease at both ends
    e = a.ease
    if e > 0:
        tgt = np.interp(u, [0, e, 1 - e, 1], [0, e / 2, 1 - e / 2, 1])
        tgt = np.where(u < e, (e / 2) * (u / e) ** 2, tgt)
        tgt = np.where(u > 1 - e, 1 - (e / 2) * ((1 - u) / e) ** 2, tgt)
    else:
        tgt = u  # constant speed edge to edge: the shot is already moving when it cuts in and out
    t_src = np.interp(tgt, prog, np.arange(len(prog)))
    if a.no_retime:
        n_out = len(prog)
        t_src = np.arange(n_out, dtype=float)

    # output 16:9 at 1920x1080
    out_w, out_h = 1920, 1080
    crop_h = h / z
    crop_w = min(w / z, crop_h * 16 / 9)
    crop_h = crop_w * 9 / 16
    Z = np.array([[out_w / crop_w, 0, -(w - crop_w) / 2 * out_w / crop_w],
                  [0, out_h / crop_h, -(h - crop_h) / 2 * out_h / crop_h], [0, 0, 1]])

    enc = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24',
                            '-s', f'{out_w}x{out_h}', '-r', str(a.fps), '-i', '-',
                            '-c:v', 'libx264', '-crf', '15', '-preset', 'slow', '-pix_fmt', 'yuv420p',
                            '-movflags', '+faststart', a.dst], stdin=subprocess.PIPE)
    cap = cv2.VideoCapture(a.src)
    cache, idx = {}, -1

    def frame(i):
        nonlocal idx
        while idx < i:
            ok, f = cap.read()
            idx += 1
            cache[idx] = cv2.warpPerspective(f, Z @ W[idx], (out_w, out_h),
                                             flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REPLICATE)
            for k in [k for k in cache if k < i - 1]:
                del cache[k]
        return cache[i]

    last = len(C) - 1
    for t in t_src:
        i0 = int(np.floor(t))
        i1 = min(i0 + 1, last)
        fr = t - i0
        f0 = frame(i0).astype(np.float32)
        f1 = frame(i1).astype(np.float32)
        # blend neighbours: sub-frame timing plus a touch of natural motion blur
        img = (1 - fr) * f0 + fr * f1
        enc.stdin.write(np.clip(img + 0.5, 0, 255).astype(np.uint8).tobytes())
    enc.stdin.close()
    enc.wait()
    print(f'{a.src}: crop-zoom {z:.3f}, {len(C)} src frames -> {n_out} out frames '
          f'({len(C) / a.fps:.1f}s -> {a.seconds:.1f}s)')


if __name__ == '__main__':
    main()
