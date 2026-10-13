"""Join final clips in order, fade in from / out to black.
Usage: python3 -I assemble.py OUT.mp4 clip1.mp4 clip2.mp4 ... [--xfade 0.6] [--fade-in 0.8] [--fade-out 1.0]
--xfade 0 gives straight cuts between shots."""
import argparse, subprocess, json
ap = argparse.ArgumentParser()
ap.add_argument('out')
ap.add_argument('clips', nargs='+')
ap.add_argument('--xfade', type=float, default=0.6, help='crossfade seconds between shots; 0 = hard cut')
ap.add_argument('--fade-in', type=float, default=0.8)
ap.add_argument('--fade-out', type=float, default=1.0)
a = ap.parse_args()


def dur(p):
    r = subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'json', p], capture_output=True, text=True)
    return float(json.loads(r.stdout)['format']['duration'])


d = [dur(c) for c in a.clips]
n = len(a.clips)
inputs = sum((['-i', c] for c in a.clips), [])
parts = [f'[{i}:v]fps=24,scale=1920:1080:flags=lanczos,setsar=1,format=yuv420p,settb=AVTB[v{i}]' for i in range(n)]
if a.xfade > 0:
    prev, t = 'v0', d[0]
    for i in range(1, n):
        off = t - a.xfade
        parts.append(f'[{prev}][v{i}]xfade=transition=fade:duration={a.xfade}:offset={off:.3f}[x{i}]')
        prev, t = f'x{i}', off + d[i]
else:
    parts.append(''.join(f'[v{i}]' for i in range(n)) + f'concat=n={n}:v=1:a=0[cat]')
    prev, t = 'cat', sum(d)
fades = []
if a.fade_in > 0:
    fades.append(f'fade=t=in:st=0:d={a.fade_in}')
if a.fade_out > 0:
    fades.append(f'fade=t=out:st={t - a.fade_out:.3f}:d={a.fade_out}')
parts.append(f'[{prev}]{",".join(fades) or "null"}[out]')
cmd = ['ffmpeg', '-v', 'error', '-y', *inputs, '-filter_complex', ';'.join(parts), '-map', '[out]',
       '-c:v', 'libx264', '-crf', '16', '-preset', 'slow', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', a.out]
subprocess.run(cmd, check=True)
print(f'{a.out}: {n} clips, {t:.1f} s')
