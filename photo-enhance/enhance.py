"""Deterministic real-estate photo pass: vertical straightening + light/colour grade.

No generative AI: every output pixel comes from the source photo via a
perspective warp, crop and tone/colour curves, so architecture, furniture
and fixtures cannot change.

Usage: python3 enhance.py INPUT_DIR OUTPUT_DIR
"""

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Geometry: find the vertical vanishing point and rotate the virtual camera
# level (pitch + roll only, no yaw), like a tripod shot with a level head.
# ---------------------------------------------------------------------------

def detect_segments(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    lines = lsd.detect(gray)[0]
    if lines is None:
        return np.zeros((0, 4), np.float32)
    return lines.reshape(-1, 4)


def segment_props(seg):
    x1, y1, x2, y2 = seg.T
    dx, dy = x2 - x1, y2 - y1
    length = np.hypot(dx, dy)
    # angle from vertical, in degrees
    ang = np.degrees(np.arctan2(np.abs(dx), np.abs(dy)))
    return length, ang


def homog_lines(seg):
    p1 = np.c_[seg[:, 0], seg[:, 1], np.ones(len(seg))]
    p2 = np.c_[seg[:, 2], seg[:, 3], np.ones(len(seg))]
    lines = np.cross(p1, p2)
    return lines / np.linalg.norm(lines[:, :2], axis=1, keepdims=True)


def vp_angle_error(seg, vp):
    """Angle (deg) between each segment and the ray from its midpoint to vp."""
    mid = np.c_[(seg[:, 0] + seg[:, 2]) / 2, (seg[:, 1] + seg[:, 3]) / 2]
    d_seg = np.c_[seg[:, 2] - seg[:, 0], seg[:, 3] - seg[:, 1]]
    if abs(vp[2]) < 1e-9:
        d_vp = np.tile(vp[:2], (len(seg), 1))
    else:
        d_vp = vp[:2] / vp[2] - mid
    cos = np.abs(np.sum(d_seg * d_vp, 1)) / (
        np.linalg.norm(d_seg, axis=1) * np.linalg.norm(d_vp, axis=1) + 1e-12)
    return np.degrees(np.arccos(np.clip(cos, 0, 1)))


def fit_vp(seg, weights):
    """Least-squares VP (homogeneous) in a conditioned frame."""
    lines = homog_lines(seg)
    A = lines * np.sqrt(weights)[:, None]
    _, _, vt = np.linalg.svd(A)
    vp = vt[-1]
    return vp / np.linalg.norm(vp)


def find_vertical_vp(img, max_tilt_deg=30, min_len_frac=0.04, seed=0):
    h, w = img.shape[:2]
    seg = detect_segments(img)
    length, ang = segment_props(seg)
    keep = (ang < max_tilt_deg) & (length > min_len_frac * max(h, w))
    seg, length = seg[keep], length[keep]
    # condition: centre + scale so homogeneous maths is stable
    s = max(h, w)
    T = np.array([[1 / s, 0, -w / 2 / s], [0, 1 / s, -h / 2 / s], [0, 0, 1]])
    segn = seg.copy()
    segn[:, [0, 1]] = (seg[:, [0, 1]] - [w / 2, h / 2]) / s
    segn[:, [2, 3]] = (seg[:, [2, 3]] - [w / 2, h / 2]) / s
    lines = homog_lines(segn)

    rng = np.random.default_rng(seed)
    best, best_score = None, -1
    for _ in range(3000):
        i, j = rng.choice(len(segn), 2, replace=False, p=length / length.sum())
        vp = np.cross(lines[i], lines[j])
        if np.linalg.norm(vp) < 1e-12:
            continue
        vp /= np.linalg.norm(vp)
        err = vp_angle_error(segn, vp)
        score = length[err < 1.0].sum()
        if score > best_score:
            best, best_score = vp, score
    inl = vp_angle_error(segn, best) < 1.5
    vp_n = fit_vp(segn[inl], length[inl])
    # back to pixel coordinates
    vp = np.linalg.inv(T) @ vp_n
    return vp, seg[inl], inl.sum(), len(seg)


def rodrigues(a, b):
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    s = np.linalg.norm(v)
    if s < 1e-12:
        return np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * ((1 - c) / s ** 2)


def level_homography(vp, K):
    d = np.linalg.inv(K) @ vp
    if d[2] < 0:
        d = -d
    target = np.array([0.0, 1.0 if d[1] > 0 else -1.0, 0.0])
    R = rodrigues(d, target)
    return K @ R @ np.linalg.inv(K), R


def rotation_angles(R):
    """Pitch / roll (deg) removed by R."""
    pitch = math.degrees(math.atan2(R[2, 1], R[2, 2]))
    roll = math.degrees(math.atan2(R[1, 0], R[0, 0]))
    return pitch, roll


def largest_rect(poly, aspect, horizon_y=None, horizon_target=0.48):
    """Best axis-aligned rectangle (w/h = aspect) inside a convex polygon.

    Maximises size, but when horizon_y is given it penalises crops that push
    the (now level) horizon far from horizon_target of the frame height, so a
    tilted-down shot doesn't end up as a floor-heavy crop.
    """
    poly = np.asarray(poly, np.float64)
    ys = np.linspace(poly[:, 1].min(), poly[:, 1].max(), 1200)

    def x_extent(y):
        xs = []
        n = len(poly)
        for k in range(n):
            (x1, y1), (x2, y2) = poly[k], poly[(k + 1) % n]
            if (y1 - y) * (y2 - y) <= 0 and y1 != y2:
                xs.append(x1 + (y - y1) * (x2 - x1) / (y2 - y1))
        if len(xs) < 2:
            return None
        return min(xs), max(xs)

    ext = [x_extent(y) for y in ys]
    L = np.array([e[0] if e else np.inf for e in ext])
    Rr = np.array([e[1] if e else -np.inf for e in ext])
    best = None
    n = len(ys)
    dy = ys[1] - ys[0]
    for i in range(n):
        if not np.isfinite(L[i]):
            continue
        # binary search the tallest rectangle starting at row i
        lo, hi = 0, n - 1 - i
        while lo < hi:
            mid = (lo + hi + 1) // 2
            j = i + mid
            left = max(L[i], L[j])
            right = min(Rr[i], Rr[j])
            if right - left >= aspect * (ys[j] - ys[i]):
                lo = mid
            else:
                hi = mid - 1
        j = i + lo
        hgt = ys[j] - ys[i]
        if hgt <= 0:
            continue
        left, right = max(L[i], L[j]), min(Rr[i], Rr[j])
        wid = aspect * hgt
        x0 = (left + right) / 2 - wid / 2
        score = hgt
        if horizon_y is not None:
            hpos = (horizon_y - ys[i]) / hgt
            score *= 1 - 1.5 * max(0.0, abs(hpos - horizon_target) - 0.04)
        cand = (score, x0, ys[i], wid, hgt)
        if best is None or score > best[0] + dy * 0.5:
            best = cand
    _, x0, y0, wid, hgt = best
    return x0, y0, wid, hgt


def straighten(img, focal_px, debug=None):
    """Return the warp M (source -> level, cropped, same size) and stats."""
    h, w = img.shape[:2]
    K = np.array([[focal_px, 0, w / 2], [0, focal_px, h / 2], [0, 0, 1]])
    vp, inliers, n_in, n_all = find_vertical_vp(img)
    H, R = level_homography(vp, K)
    pitch, roll = rotation_angles(R)

    corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float64)
    wc = cv2.perspectiveTransform(corners[None], H)[0]
    # a level camera's horizon sits on the principal-point row (K R K^-1)
    x0, y0, cw, ch = largest_rect(wc, w / h, horizon_y=h / 2)
    # map crop rectangle to output size w x h directly (one resample)
    S = np.array([[w / cw, 0, -x0 * w / cw], [0, h / ch, -y0 * h / ch], [0, 0, 1]])
    M = S @ H
    # share of the original frame that survives the crop
    rect = np.array([[x0, y0], [x0 + cw, y0], [x0 + cw, y0 + ch], [x0, y0 + ch]])
    back = cv2.perspectiveTransform(rect[None], np.linalg.inv(H))[0]
    kept = cv2.contourArea(back.astype(np.float32)) / (w * h)
    info = dict(pitch_deg=round(pitch, 2), roll_deg=round(roll, 2),
                vertical_lines=int(n_in), candidate_lines=int(n_all),
                original_frame_kept=round(float(kept), 3),
                upsample=round(float(w / cw), 3))
    if debug is not None:
        dbg = img.copy()
        for x1, y1, x2, y2 in inliers.astype(int):
            cv2.line(dbg, (x1, y1), (x2, y2), (0, 0, 255), 2)
        debug["lines"] = dbg
    return M, info


# ---------------------------------------------------------------------------
# Light + colour grade. Works in linear light; only per-pixel curves and an
# edge-aware tone map, so nothing is added, removed or moved.
# ---------------------------------------------------------------------------

def srgb_to_lin(x):
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def lin_to_srgb(x):
    x = np.clip(x, 0, None)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1 / 2.4) - 0.055)


def guided_filter(I, p, r, eps):
    """He et al. guided filter (grey guide), box filters only."""
    def box(x):
        return cv2.boxFilter(x, -1, (2 * r + 1, 2 * r + 1), borderType=cv2.BORDER_REFLECT)
    mI, mp = box(I), box(p)
    cov = box(I * p) - mI * mp
    var = box(I * I) - mI * mI
    a = cov / (var + eps)
    b = mp - a * mI
    return box(a) * I + box(b)


def luminance(lin):
    return 0.2126 * lin[..., 0] + 0.7152 * lin[..., 1] + 0.0722 * lin[..., 2]


def white_balance(lin, ref_box, target, strength=1.0):
    if ref_box is None or strength == 0:
        return lin, (1.0, 1.0, 1.0)
    x0, y0, x1, y1 = ref_box
    ref = lin[y0:y1, x0:x1].reshape(-1, 3).mean(0)
    gr = (target[0] * ref[1] / ref[0]) ** strength
    gb = (target[1] * ref[1] / ref[2]) ** strength
    gains = np.array([gr, 1.0, gb], np.float32)
    # keep overall brightness: normalise so luminance of grey is unchanged
    gains /= (0.2126 * gr + 0.7152 + 0.0722 * gb)
    return lin * gains, tuple(float(g) for g in gains)


def tone_map(lin, mid_target=0.34, compress=0.66, detail=1.08, shoulder=0.75,
             max_white=1.8):
    """Lift shadows / balance exposure with a base-detail split in log space.

    compress < 1 shrinks the large-scale brightness range (darker regions
    come up more than bright ones); detail > 1 keeps texture crisp so it
    doesn't go flat. Exposure puts the median at mid_target (linear) unless
    that would push the 97th percentile past max_white before the shoulder.
    Values are kept modest to avoid the HDR look.
    """
    Y = np.maximum(luminance(lin), 1e-5)
    logY = np.log2(Y).astype(np.float32)
    h, w = Y.shape
    r = max(8, int(max(h, w) / 40))
    guide = (logY - logY.min()) / (np.ptp(logY) + 1e-6)
    base = guided_filter(guide, logY, r, 0.01)
    det = logY - base
    hi = np.percentile(base, 99)
    newlog = hi + (base - hi) * compress + det * detail
    ev = min(math.log2(mid_target) - np.median(newlog),
             math.log2(max_white) - np.percentile(newlog, 97))
    newY = np.exp2(newlog + ev)
    # soft shoulder so windows and sky roll off instead of clipping
    k = shoulder
    newY = np.where(newY > k, k + (1 - k) * np.tanh((newY - k) / (1 - k)), newY)
    out = lin * (newY / Y)[..., None]
    # a saturated colour can push one channel past 1: scale it back (keeps
    # hue) and blend toward white by the overshoot (keeps brightness)
    m = np.maximum(out.max(axis=2, keepdims=True), 1.0)
    t = (m - 1.0) / m
    out = (out / m) * (1 - t) + t
    return np.clip(out, 0, 1), float(ev)


def contrast_curve(srgb, black=0.012, strength=0.10):
    """Gentle S-curve + small black point so lifted shadows keep depth."""
    x = np.clip((srgb - black) / (1 - black), 0, 1)
    s = x + strength * x * (1 - x) * (2 * x - 1) * 2  # zero at 0, 0.5, 1
    return np.clip(s, 0, 1)


def colour(srgb, vibrance=0.10, wood=0.08):
    hsv = cv2.cvtColor(srgb.astype(np.float32), cv2.COLOR_RGB2HSV)
    hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    # vibrance: lift low-saturation colours more than strong ones
    sat = sat + vibrance * sat * (1 - sat)
    # warm wood tones (orange/brown band ~15-45 deg) get a little extra
    wood_w = np.clip(1 - np.abs(hue - 28) / 18, 0, 1) * np.clip(sat * 3, 0, 1)
    sat = sat * (1 + wood * wood_w)
    hsv = np.dstack([hue, np.clip(sat, 0, 1), val])
    return cv2.cvtColor(hsv.astype(np.float32), cv2.COLOR_HSV2RGB)


def sharpen(srgb, radius=1.1, amount=0.55, threshold=0.012):
    lab = cv2.cvtColor(srgb.astype(np.float32), cv2.COLOR_RGB2Lab)
    L = lab[..., 0] / 100.0
    blur = cv2.GaussianBlur(L, (0, 0), radius)
    d = L - blur
    d = np.where(np.abs(d) < threshold, 0, d)
    lab[..., 0] = np.clip(L + amount * d, 0, 1) * 100
    return np.clip(cv2.cvtColor(lab, cv2.COLOR_Lab2RGB), 0, 1)


# ---------------------------------------------------------------------------
# Per-photo settings. ref = a patch of known-white surface (x0, y0, x1, y1 in
# source pixels) used for white balance; None skips white balance.
# ---------------------------------------------------------------------------

WARM_WHITE = (1.02, 0.92)  # linear R/G, B/G: clean white with a touch of warmth

DEFAULTS = dict(ref=None, wb_target=WARM_WHITE, wb_strength=1.0, focal_frac=0.32,
                mid_target=0.34, compress=0.66, detail=1.08,
                black=0.012, contrast=0.10, vibrance=0.10, wood=0.08)

PHOTOS = {
    "1": dict(ref=(300, 100, 900, 300)),             # living room: white ceiling
    "2": dict(ref=(400, 100, 900, 300)),             # dining nook: ceiling
    "3": dict(ref=(600, 600, 900, 700), wb_target=(1.02, 0.93)),  # bath: wall
    "4": dict(ref=(850, 650, 1050, 850)),            # arch room: sunlit wall
    "5": dict(ref=None, mid_target=0.21, compress=0.75, vibrance=0.12,
              wood=0.0),                             # exterior: overcast
}


def process(path, cfg):
    src = cv2.imread(str(path))
    h, w = src.shape[:2]
    # light JPEG-artefact cleanup before shadows are lifted
    img = cv2.fastNlMeansDenoisingColored(src, None, 2, 2, 7, 21)
    rgb = img[..., ::-1].astype(np.float32) / 255.0
    lin = srgb_to_lin(rgb).astype(np.float32)
    lin, gains = white_balance(lin, cfg["ref"], cfg["wb_target"], cfg["wb_strength"])
    lin, ev = tone_map(lin, cfg["mid_target"], cfg["compress"], cfg["detail"])
    srgb = lin_to_srgb(lin).astype(np.float32)
    srgb = contrast_curve(srgb, cfg["black"], cfg["contrast"])
    srgb = colour(srgb, cfg["vibrance"], cfg["wood"])

    # geometry is measured on the untouched source, applied to the graded image
    diag = math.hypot(max(w, h), max(w, h) * 0.75)  # full 4:3 sensor frame
    M, ginfo = straighten(src, cfg["focal_frac"] * diag)
    warped = cv2.warpPerspective(srgb, M, (w, h), flags=cv2.INTER_LANCZOS4,
                                 borderMode=cv2.BORDER_REFLECT)
    out = sharpen(np.clip(warped, 0, 1))
    info = dict(ginfo, wb_gains=[round(g, 3) for g in gains], exposure_ev=round(ev, 2))
    return (out[..., ::-1] * 255 + 0.5).astype(np.uint8), info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input_dir", type=Path)
    ap.add_argument("output_dir", type=Path)
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {}
    for path in sorted(args.input_dir.glob("*.jpg")):
        cfg = dict(DEFAULTS, **PHOTOS.get(path.stem, {}))
        out, info = process(path, cfg)
        cv2.imwrite(str(args.output_dir / f"{path.stem}_enhanced.jpg"), out,
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
        report[path.name] = info
        print(path.name, info)
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
