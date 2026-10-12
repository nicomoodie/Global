"""Bounce/sway QC that is not fooled by zoom, drift or speed ramps.

Per frame: global similarity (LK + RANSAC, like motion.py) composed properly.
Translation is measured at the image centre, then the part explained by a
zoom about ONE fixed expansion point (= a straight rail in any direction) is
removed by least squares. What is left is camera-path deviation from a rail.
  sway_px / bob_px : peak |deviation| in the gait band (>~0.6 Hz), interior frames
  rail_dev_px      : peak-to-peak deviation from the rail without high-pass
                     (catches slow weaves and mid-clip height humps)
  roll_deg         : peak |roll| (horizon tilt) relative to frame 0
Usage: python3 -I motion_qc.py clip.mp4 [...]
"""
import sys, json
import cv2, numpy as np


def steps(path):
    cap = cv2.VideoCapture(path)
    ok, prev = cap.read()
    h, w = prev.shape[:2]
    pg = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    A = [np.eye(3)]
    while True:
        ok, f = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        p = cv2.goodFeaturesToTrack(pg, 600, 0.01, 15)
        q, st, _ = cv2.calcOpticalFlowPyrLK(pg, g, p, None)
        M, _ = cv2.estimateAffinePartial2D(p[st == 1], q[st == 1], method=cv2.RANSAC,
                                           ransacReprojThreshold=2.0)
        A.append(np.vstack([M, [0, 0, 1]]))
        pg = g
    return A, w, h


def hp(x, sigma=10.0):
    """x minus a Gaussian trend; odd-reflect padding so ends keep their slope."""
    r = int(3 * sigma)
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2); k /= k.sum()
    xp = np.concatenate([2 * x[0] - x[r:0:-1], x, 2 * x[-1] - x[-2:-r - 2:-1]])
    return x - np.convolve(xp, k, mode='valid')


def analyse(path, fps=24):
    A, w, h = steps(path)
    C = [A[0]]
    for M in A[1:]:
        C.append(M @ C[-1])
    C = np.array(C)
    c = np.array([w / 2, h / 2, 1.0])
    s = np.hypot(C[:, 0, 0], C[:, 1, 0])
    th = np.arctan2(C[:, 1, 0], C[:, 0, 0])
    u = (C @ c)[:, :2] - c[:2]                 # where the frame centre went
    e = (s - 1)[:, None]
    g = (e * u).sum(0) / max((e * e).sum(), 1e-12)   # u = (s-1)*(c-F): fit fixed F
    F = c[:2] - g
    r = u - e * g                              # deviation from a straight rail
    m = slice(6, len(r) - 6)                   # drop 1/4 s at each end
    out = {}
    for i, name in ((0, 'sway_px'), (1, 'bob_px')):
        j = hp(r[:, i])[m]
        sp = np.abs(np.fft.rfft(j - j.mean())); fr = np.fft.rfftfreq(len(j), 1 / fps)
        out[name] = round(float(np.abs(j).max()), 2)
        out[name + '_hz'] = round(float(fr[1:][sp[1:].argmax()]), 2)
        out['rail_dev_' + name[:-3] + '_pp'] = round(float(np.ptp(r[m, i])), 2)
    out['roll_deg_peak'] = round(float(np.degrees(np.abs(th).max())), 3)
    out['zoom_total'] = round(float(s[-1]), 4)
    out['expansion_point_frac'] = [round(float(F[0] / w), 3), round(float(F[1] / h), 3)]
    ls = np.log(s)
    v = np.diff(ls) * fps
    out['speed_cv_mid'] = round(float(v[len(v)//5:-len(v)//5].std() / max(abs(v[len(v)//5:-len(v)//5].mean()), 1e-9)), 3)
    return out


if __name__ == '__main__':
    print(json.dumps({p.split('/')[-1]: analyse(p) for p in sys.argv[1:]}, indent=1))
