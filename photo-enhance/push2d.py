"""Local, perfectly steady push-in on a still (for planar exteriors). No AI, nothing invented.
Usage: python3 -I push2d.py IN.png OUT.mp4 --aim X Y --z 1.07 [--seconds 5]"""
import argparse, subprocess
import cv2, numpy as np
ap = argparse.ArgumentParser()
ap.add_argument('src'); ap.add_argument('dst')
ap.add_argument('--aim', type=float, nargs=2, required=True, help='expansion point in source px')
ap.add_argument('--z', type=float, required=True); ap.add_argument('--seconds', type=float, default=5)
ap.add_argument('--fps', type=int, default=24)
a = ap.parse_args()
img = cv2.imread(a.src); h, w = img.shape[:2]
OW, OH = 1920, 1080
n = int(round(a.seconds * a.fps)); fx, fy = a.aim
enc = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{OW}x{OH}',
                        '-r', str(a.fps), '-i', '-', '-c:v', 'libx264', '-crf', '15', '-preset', 'slow',
                        '-pix_fmt', 'yuv420p', '-movflags', '+faststart', a.dst], stdin=subprocess.PIPE)
for i in range(n):
    u = i / (n - 1)
    e = 0.15  # constant speed with soft ease at both ends
    t = np.interp(u, [0, e, 1 - e, 1], [0, e / 2, 1 - e / 2, 1])
    if u < e: t = (e / 2) * (u / e) ** 2
    if u > 1 - e: t = 1 - (e / 2) * ((1 - u) / e) ** 2
    s = np.exp(np.log(a.z) * t)           # geometric zoom = constant apparent speed
    # source crop of size (w/s, h/s) scaled about the aim point, mapped to the output frame
    M = np.array([[s * OW / w, 0, (fx - s * fx) * OW / w], [0, s * OH / h, (fy - s * fy) * OH / h]])
    enc.stdin.write(cv2.warpAffine(img, M, (OW, OH), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT).tobytes())
enc.stdin.close(); enc.wait(); print(a.dst, n, 'frames')
