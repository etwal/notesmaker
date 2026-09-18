"""
Reconstruct a scrolling-whiteboard lecture recording (Panopto screen capture)
into a set of page images + a PDF.

Usage:
    python extract_notes.py lecture.mp4                # -> lecture_notes.pdf + lecture_notes/page_XX.png
    python extract_notes.py lecture.mp4 --out sep23    # -> sep23.pdf + sep23/page_XX.png

Requires: ffmpeg on PATH, and `pip install opencv-python numpy pillow`.
Takes roughly 10-15 min for a 55-min lecture.
"""
import argparse, subprocess, json, os, sys
import numpy as np, cv2
from PIL import Image

W, H = 1280, 720
X0, X1, Y0, Y1 = 44, 1236, 72, 716      # notes region: excludes toolbar strip + dark side borders
RW, RH = X1 - X0, Y1 - Y0
PAD_Y, PAD_X = 12000, 600
SC = 4

def frames(vid, fps):
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-i", vid, "-vf", f"fps={fps}",
                          "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], stdout=subprocess.PIPE)
    i = 0
    while True:
        buf = p.stdout.read(W * H * 3)
        if len(buf) < W * H * 3: break
        yield round(i / fps, 2), np.frombuffer(buf, np.uint8).reshape(H, W, 3); i += 1
    p.kill()

def ink(g): return (255 - g).astype(np.float32)

def track(vid, fps):
    """Pass 1: find each frame's position on the global canvas."""
    CH, CW = RH + 2 * PAD_Y, RW + 2 * PAD_X
    canvas = np.full((CH, CW), 255, np.uint8); written = np.zeros((CH, CW), bool)
    px, py = PAD_X, PAD_Y; prevg = None; pos = {}
    for t, f in frames(vid, fps):
        g = cv2.cvtColor(f[Y0:Y1, X0:X1], cv2.COLOR_BGR2GRAY)
        if g.mean() < 40: continue                                    # black frame (recording ended)
        if prevg is not None:                                         # predict via phase correlation
            (dx, dy), resp = cv2.phaseCorrelate(prevg.astype(np.float32), g.astype(np.float32))
            if resp >= 0.05 and (abs(dx) >= 1 or abs(dy) >= 1):
                px -= int(round(dx)); py -= int(round(dy))
        fi = ink(g)
        if (fi > 90).sum() >= 400:                                    # refine against canvas built so far
            SR = 220
            y0, x0 = max(0, py - SR), max(0, px - SR // 4)
            y1, x1 = min(CH, py + RH + SR), min(CW, px + RW + SR // 4)
            if written[y0:y1, x0:x1].any():
                ci = cv2.resize(ink(canvas[y0:y1, x0:x1]), None, fx=1/SC, fy=1/SC, interpolation=cv2.INTER_AREA)
                ti = cv2.resize(fi, None, fx=1/SC, fy=1/SC, interpolation=cv2.INTER_AREA)
                _, mx, _, loc = cv2.minMaxLoc(cv2.matchTemplate(ci, ti, cv2.TM_CCOEFF_NORMED))
                if mx >= 0.35:
                    cx, cy = x0 + loc[0] * SC - SC, y0 + loc[1] * SC - SC
                    r = cv2.matchTemplate(ink(canvas[cy:cy + RH + 2*SC, cx:cx + RW + 2*SC]), fi, cv2.TM_CCOEFF_NORMED)
                    _, _, _, loc = cv2.minMaxLoc(r)
                    px, py = cx + loc[0], cy + loc[1]
        pos[t] = (px, py)
        canvas[py:py + RH, px:px + RW] = g; written[py:py + RH, px:px + RW] = True; prevg = g
        if int(t) % 300 == 0 and t == int(t): print(f"  tracked {int(t)//60} min", flush=True)
    return pos

def compose(vid, fps, pos):
    """Pass 2: paste only stationary frames (mid-scroll frames are smeared by the encoder)."""
    ts = sorted(pos)
    minx = min(p[0] for p in pos.values()); miny = min(p[1] for p in pos.values())
    maxx = max(p[0] for p in pos.values()); maxy = max(p[1] for p in pos.values())
    canvas = np.full((maxy - miny + RH, maxx - minx + RW, 3), 255, np.uint8)
    still = set()
    for i, t in enumerate(ts):
        p = pos[t]
        if all(abs(pos[ts[j]][0] - p[0]) <= 1 and abs(pos[ts[j]][1] - p[1]) <= 1
               for j in (i - 1, i + 1) if 0 <= j < len(ts)):
            still.add(t)
    for t, f in frames(vid, fps):
        if t not in still: continue
        x, y = pos[t][0] - minx, pos[t][1] - miny
        canvas[y:y + RH, x:x + RW] = f[Y0:Y1, X0:X1]
    return canvas

def paginate(c):
    """Split the tall canvas into letter-proportioned pages, cutting only at blank gaps."""
    g = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY); inkrow = (g < 160).sum(axis=1)
    rows = np.where(inkrow > 0)[0]
    c = c[max(rows[0] - 40, 0):min(rows[-1] + 60, c.shape[0])]; inkrow = inkrow[max(rows[0] - 40, 0):min(rows[-1] + 60, len(inkrow))]
    Hc, Wc = c.shape[:2]; target = int(Wc * 11 / 8.5); pages = []; y = 0
    while y < Hc:
        end = min(y + target, Hc)
        if end < Hc:
            for e in range(end, max(y + target - 300, y + 200), -1):
                if inkrow[e - 12:e].sum() == 0: end = e - 6; break
        page = np.full((target, Wc, 3), 255, np.uint8); page[:end - y] = c[y:end]
        pages.append(page); y = end
    return pages

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("video"); ap.add_argument("--out", help="output basename (default: <video>_notes)")
    ap.add_argument("--fps", type=int, default=3)
    a = ap.parse_args()
    out = a.out or os.path.splitext(os.path.basename(a.video))[0] + "_notes"
    print("Pass 1/2: tracking scroll position ..."); pos = track(a.video, a.fps)
    print("Pass 2/2: compositing ..."); canvas = compose(a.video, a.fps, pos)
    os.makedirs(out, exist_ok=True); cv2.imwrite(os.path.join(out, "full_canvas.png"), canvas)
    pages = paginate(canvas); ims = []
    for i, p in enumerate(pages, 1):
        cv2.imwrite(os.path.join(out, f"page_{i:02d}.png"), p)
        ims.append(Image.fromarray(cv2.cvtColor(p, cv2.COLOR_BGR2RGB)))
    ims[0].save(out + ".pdf", save_all=True, append_images=ims[1:], resolution=150)
    print(f"Done: {len(pages)} pages -> {out}.pdf and {out}/")
