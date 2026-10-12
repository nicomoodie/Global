"""Join final clips in order with crossfades, fade in/out. Usage: python3 -I assemble.py OUT.mp4 clip1.mp4 clip2.mp4 ..."""
import subprocess, sys, json
out, clips = sys.argv[1], sys.argv[2:]
XF, FIN, FOUT = 0.6, 0.8, 1.0
def dur(p):
    r = subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'json', p], capture_output=True, text=True)
    return float(json.loads(r.stdout)['format']['duration'])
d = [dur(c) for c in clips]
inputs = sum((['-i', c] for c in clips), [])
parts = [f'[{i}:v]fps=24,scale=1920:1080:flags=lanczos,setsar=1,format=yuv420p,settb=AVTB[v{i}]' for i in range(len(clips))]
prev, t = 'v0', d[0]
for i in range(1, len(clips)):
    off = t - XF
    parts.append(f'[{prev}][v{i}]xfade=transition=fade:duration={XF}:offset={off:.3f}[x{i}]')
    prev, t = f'x{i}', off + d[i]
parts.append(f'[{prev}]fade=t=in:st=0:d={FIN},fade=t=out:st={t - FOUT:.3f}:d={FOUT}[out]')
cmd = ['ffmpeg', '-v', 'error', '-y', *inputs, '-filter_complex', ';'.join(parts), '-map', '[out]',
       '-c:v', 'libx264', '-crf', '16', '-preset', 'slow', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', out]
subprocess.run(cmd, check=True)
print(f'{out}: {len(clips)} clips, {t:.1f} s')
