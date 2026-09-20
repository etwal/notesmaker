"""
Reconstruct a scrolling-whiteboard lecture recording (Panopto screen capture)
into a set of page images + a PDF.

Usage:
    python extract_notes.py lecture.mp4                # -> lecture_notes.pdf + lecture_notes/page_XX.png
    python extract_notes.py lecture.mp4 --out sep23    # -> sep23.pdf + sep23/page_XX.png

Requires: ffmpeg on PATH, and `pip install opencv-python numpy pillow`.
Takes roughly 10-15 min for a 55-min lecture.
"""
import argparse, subprocess, json, os
import numpy as np, cv2
from PIL import Image

W, H = 1280, 720
X0, X1, Y0, Y1 = 44, 1220, 84, 692      # notes region: excludes toolbar (white box to row ~78), scrollbars, dark side borders
RW, RH = X1 - X0, Y1 - Y0
PAD_Y, PAD_X = 40000, 2000                # room for vertical scrolling, horizontal panning and re-seeded areas
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

def is_whiteboard(f):
    """Fullscreen OneNote layout: black side borders, white notes area."""
    if f[:, :40].mean() > 25 or f[:, 1245:].mean() > 25 or f[700:, 100:1200].mean() < 225: return False
    if f[100:700, 44:70].mean() < 200 or f[100:700, 1195:1220].mean() < 200: return False   # notes area must reach the crop edges
    g = cv2.cvtColor(f[Y0:Y1, X0:X1], cv2.COLOR_BGR2GRAY)
    return (g > 238).mean() > 0.85

def overlay_mask(reg):
    """Popup menus / on-screen keyboard / mini toolbars. Two signatures:
    (a) boxes drawn with long thin light-grey lines, (b) dense clusters of tiny grey glyphs
    (menu text, key labels) - handwriting is large strokes in black or saturated colour.
    Returns a bool mask (True = keep) covering everything outside the overlay's box."""
    hsv = cv2.cvtColor(reg, cv2.COLOR_BGR2HSV); g = cv2.cvtColor(reg, cv2.COLOR_BGR2GRAY)
    keep = np.ones(g.shape, bool); P = 25
    m = ((g > 140) & (g < 232) & (hsv[..., 1] < 40)).astype(np.uint8)
    strong = ((g < 140) | (hsv[..., 1] >= 60)).astype(np.uint8)       # real ink; its anti-aliased fringe is grey too
    m[cv2.dilate(strong, np.ones((5, 5), np.uint8)) > 0] = 0
    hk, vk = np.ones((1, 50), np.uint8), np.ones((50, 1), np.uint8)
    lines = cv2.morphologyEx(m, cv2.MORPH_OPEN, hk) | cv2.morphologyEx(m, cv2.MORPH_OPEN, vk)
    if lines.sum() >= 200:
        ys, xs = np.where(lines)
        keep[max(ys.min() - P, 0):ys.max() + P, max(xs.min() - P, 0):xs.max() + P] = False
    m = ((g < 225) & (hsv[..., 1] < 45)).astype(np.uint8)
    n, lab, stats, cent = cv2.connectedComponentsWithStats(m, 8)
    w, h, a = stats[1:, cv2.CC_STAT_WIDTH], stats[1:, cv2.CC_STAT_HEIGHT], stats[1:, cv2.CC_STAT_AREA]
    small = (w >= 2) & (w <= 14) & (h >= 2) & (h <= 14) & (a >= 4)
    if small.sum() >= 25:                                             # UI glyphs are thin: no solid-black core, unlike pen strokes
        dark = (g < 60).astype(np.uint8)
        idx = np.flatnonzero(small) + 1
        small[idx - 1] = np.array([dark[lab == i].mean() < 0.3 for i in idx])
    if small.sum() >= 25:
        pts = np.zeros(g.shape, np.uint8)
        for cx, cy in cent[1:][small]: pts[int(cy), int(cx)] = 1
        pts = cv2.dilate(pts, np.ones((81, 81), np.uint8))          # link glyphs within ~40 px
        nc, cl, cst, _ = cv2.connectedComponentsWithStats(pts, 8)
        for i in range(1, nc):
            x, y, cw, ch = cst[i, :4]
            cnt = sum(1 for cx, cy in cent[1:][small] if cl[int(cy), int(cx)] == i)
            if cnt >= 12:
                keep[max(y - P, 0):y + ch + P, max(x - P, 0):x + cw + P] = False
    return keep

def track(vid, fps):
    """Pass 1: find each frame's position (and zoom scale) on the global canvas."""
    CH, CW = RH + 2 * PAD_Y, RW + 2 * PAD_X
    canvas = np.full((CH, CW), 255, np.uint8)
    written = np.zeros((CH // SC + 1, CW // SC + 1), bool)          # kept at 1/SC scale to save memory
    count = np.zeros_like(written, np.uint16)                        # how many frames were pasted on each spot
    px, py, scale = PAD_X, PAD_Y, 1.0; prevg = None; log = {}; lost_run = 0
    def scaled(img, sc, interp=cv2.INTER_AREA):
        return img if sc == 1.0 else cv2.resize(img, None, fx=sc, fy=sc, interpolation=interp)
    def coarse(ti, y0, x0, y1, x1, confirmed=False, exclude=None):
        """Best match of template ti in the canvas window; the score is zeroed unless the canvas
        under the match holds a good share of the template's ink (blank-on-blank scores high).
        confirmed=True ignores canvas areas seen by fewer than 5 s of frames (stray islands),
        and exclude=(y0,x0,y1,x1) blanks that area (the island we are trying to leave)."""
        ci = cv2.resize(ink(canvas[y0:y1, x0:x1]), None, fx=1/SC, fy=1/SC, interpolation=cv2.INTER_AREA)
        if confirmed:
            cs = count[y0 // SC:y0 // SC + ci.shape[0], x0 // SC:x0 // SC + ci.shape[1]]
            hh, ww = min(cs.shape[0], ci.shape[0]), min(cs.shape[1], ci.shape[1])
            ci = ci[:hh, :ww].copy(); ci[cs[:hh, :ww] < 5 * fps] = 0
        if exclude:
            ey0, ex0, ey1, ex1 = exclude
            ci[max(0, (ey0 - y0) // SC):(ey1 - y0) // SC + 1, max(0, (ex0 - x0) // SC):(ex1 - x0) // SC + 1] = 0
        if ci.shape[0] < ti.shape[0] or ci.shape[1] < ti.shape[1]: return 0, None
        _, mx, _, loc = cv2.minMaxLoc(cv2.matchTemplate(ci, ti, cv2.TM_CCOEFF_NORMED))
        under = ci[loc[1]:loc[1] + ti.shape[0], loc[0]:loc[0] + ti.shape[1]]
        if (under > 60).sum() < 0.3 * (ti > 60).sum(): mx = 0.0
        return mx, (x0 + loc[0] * SC, y0 + loc[1] * SC)
    def whole():
        ys, xs = np.where(written)
        return ys.min() * SC, xs.min() * SC, min(CH, (ys.max() + 1) * SC), min(CW, (xs.max() + 1) * SC)
    def zoom_search(g0, scale, exclude=None):
        """Match the frame against the confirmed canvas over a range of zoom factors."""
        best = (0.0, None, scale)
        for zf in np.concatenate([[1.0], np.arange(0.72, 0.99, 0.03), np.arange(1.03, 1.40, 0.03)]):
            sc2 = round(float(scale * zf), 3)
            ti2 = cv2.resize(ink(scaled(g0, sc2)), None, fx=1/SC, fy=1/SC, interpolation=cv2.INTER_AREA)
            mx2, c2 = coarse(ti2, *whole(), confirmed=True, exclude=exclude)
            if mx2 > best[0]: best = (mx2, c2, sc2)
        return best
    island = None            # a freshly re-seeded area we keep trying to merge back into the main canvas
    for t, f in frames(vid, fps):
        if not is_whiteboard(f): continue
        reg = f[Y0:Y1, X0:X1]; keep = overlay_mask(reg)
        g0 = cv2.cvtColor(reg, cv2.COLOR_BGR2GRAY); g0[~keep] = 255
        moving = False
        if prevg is not None:                                         # predict via phase correlation
            (dx, dy), resp = cv2.phaseCorrelate(prevg.astype(np.float32), g0.astype(np.float32))
            if resp >= 0.05 and (abs(dx) >= 1 or abs(dy) >= 1):
                px -= int(round(dx * scale)); py -= int(round(dy * scale)); moving = True
        g = scaled(g0, scale); keep_s = scaled(keep.astype(np.uint8), scale, cv2.INTER_NEAREST).astype(bool)
        h, w = g.shape
        fi = ink(g); score = 1.0; status = "lowink"; found = None
        if (fi > 90).sum() >= 400 * scale * scale:
            ti = cv2.resize(fi, None, fx=1/SC, fy=1/SC, interpolation=cv2.INTER_AREA)
            SR = 400
            y0, x0 = max(0, py - SR), max(0, px - SR)
            y1, x1 = min(CH, py + h + SR), min(CW, px + w + SR)
            status = "blankcanvas"
            if written[y0 // SC:y1 // SC + 1, x0 // SC:x1 // SC + 1].any():
                mx, c = coarse(ti, y0, x0, y1, x1)
                if mx < 0.5:                                          # lost: search the whole canvas
                    mx2, c2 = coarse(ti, *whole(), confirmed=True)
                    if mx2 > mx: mx, c = mx2, c2
                score = mx; status = "ok" if mx >= 0.5 else "lost"
                if status == "lost" and not moving and lost_run >= 2 * fps:
                    found = zoom_search(g0, scale)                    # maybe the lecturer zoomed
                    if found[0] < 0.6:
                        # re-seed in a fresh area below everything written so far (new page?)
                        ys, xs = np.where(written); px, py = PAD_X, min((ys.max() + 1) * SC + 400, CH - h - 1)
                        status = "newarea"; found = None
                        island = {"ts": [], "box": [py, px, py + h, px + w], "tries": 0}
                elif island and not moving and status == "ok" and len(island["ts"]) % (2 * fps) == 0:
                    # on an island: periodically try to merge back into the confirmed canvas
                    cand = zoom_search(g0, scale, exclude=island["box"])
                    if cand[0] >= 0.6:
                        found = cand
                        by0, bx0, by1, bx1 = island["box"]           # discard the island
                        canvas[by0:by1, bx0:bx1] = 255
                        written[by0 // SC:by1 // SC + 1, bx0 // SC:bx1 // SC + 1] = False
                        count[by0 // SC:by1 // SC + 1, bx0 // SC:bx1 // SC + 1] = 0
                        for it in island["ts"]: log[it] = log[it][:3] + ("lost",) + log[it][4:]
                        island = None
                if found:
                    score, c, scale = found; status = "ok"
                    g = scaled(g0, scale); keep_s = scaled(keep.astype(np.uint8), scale, cv2.INTER_NEAREST).astype(bool)
                    h, w = g.shape; fi = ink(g)
                if status == "ok":
                    cx, cy = max(c[0] - SC, 0), max(c[1] - SC, 0)
                    r = cv2.matchTemplate(ink(canvas[cy:cy + h + 2*SC, cx:cx + w + 2*SC]), fi, cv2.TM_CCOEFF_NORMED)
                    _, _, _, loc = cv2.minMaxLoc(r)
                    px, py = cx + loc[0], cy + loc[1]
        log[t] = (int(px), int(py), round(float(score), 2), status, round(scale, 3))
        lost_run = lost_run + 1 if status == "lost" else 0
        if status not in ("lost", "lowink"):
            top = int(30 * scale)                                     # rows at the frame edges may show half-hidden letters
            keep_s[:top] &= canvas[py:py + top, px:px + w] > 245      # (toolbar above, scrollbar below): add ink only
            keep_s[h - top:] &= canvas[py + h - top:py + h, px:px + w] > 245
            canvas[py:py + h, px:px + w][keep_s] = g[keep_s]; written[py // SC:(py + h) // SC + 1, px // SC:(px + w) // SC + 1] = True
            count[py // SC:(py + h) // SC + 1, px // SC:(px + w) // SC + 1] += 1
            if island:
                island["ts"].append(t); b = island["box"]
                island["box"] = [min(b[0], py), min(b[1], px), max(b[2], py + h), max(b[3], px + w)]
                if len(island["ts"]) > 60 * fps: island = None        # a genuine new page: keep it
        prevg = g0
        if int(t) % 300 == 0 and t == int(t): print(f"  tracked {int(t)//60} min", flush=True)
    # label connected regions of the canvas; frames re-seeded during a fast scroll through blank
    # space form tiny isolated islands, which compose() discards
    small = cv2.dilate(written.astype(np.uint8), np.ones((25, 25), np.uint8))
    _, lab = cv2.connectedComponents(small)
    return {t: v + (int(lab[v[1] // SC, v[0] // SC]),) for t, v in log.items()}

def compose(vid, fps, log):
    """Pass 2: paste only stationary, confidently-registered frames (mid-scroll frames are smeared)."""
    from collections import Counter
    region_frames = Counter(v[5] for v in log.values() if v[3] not in ("lost", "lowink"))
    pos = {t: v for t, v in log.items()
           if v[3] not in ("lost", "lowink") and region_frames[v[5]] >= 10 * fps}
    ts = sorted(pos)
    minx = min(v[0] for v in pos.values()); miny = min(v[1] for v in pos.values())
    maxx = max(v[0] + RW * v[4] for v in pos.values()); maxy = max(v[1] + RH * v[4] for v in pos.values())
    canvas = np.full((int(maxy - miny) + 2, int(maxx - minx) + 2, 3), 255, np.uint8)
    still = set()
    for i, t in enumerate(ts):
        p = pos[t]
        if all(abs(pos[ts[j]][0] - p[0]) <= 1 and abs(pos[ts[j]][1] - p[1]) <= 1 and pos[ts[j]][4] == p[4]
               for j in (i - 1, i + 1) if 0 <= j < len(ts)):
            still.add(t)
    for t, f in frames(vid, fps):
        if t not in still: continue
        reg = f[Y0:Y1, X0:X1]; keep = overlay_mask(reg); sc = pos[t][4]
        if sc != 1.0:
            reg = cv2.resize(reg, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
            keep = cv2.resize(keep.astype(np.uint8), None, fx=sc, fy=sc, interpolation=cv2.INTER_NEAREST).astype(bool)
        h, w = keep.shape
        x, y = pos[t][0] - minx, pos[t][1] - miny
        top = int(30 * sc)
        keep[:top] &= cv2.cvtColor(canvas[y:y + top, x:x + w], cv2.COLOR_BGR2GRAY) > 245
        keep[h - top:] &= cv2.cvtColor(canvas[y + h - top:y + h, x:x + w], cv2.COLOR_BGR2GRAY) > 245
        canvas[y:y + h, x:x + w][keep] = reg[keep]
    return canvas

def paginate(c):
    """Split the tall canvas into letter-proportioned pages, cutting only at blank gaps."""
    g = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY); inkrow = (g < 160).sum(axis=1)
    rows = np.where(inkrow > 0)[0]; cols = np.where((g < 160).sum(axis=0) > 0)[0]
    lo, hi = max(rows[0] - 40, 0), min(rows[-1] + 60, c.shape[0])
    c = c[lo:hi, max(cols[0] - 40, 0):min(cols[-1] + 40, c.shape[1])]; inkrow = inkrow[lo:hi]
    keep_rows = np.ones(len(inkrow), bool); y = 0                   # collapse blank gaps longer than 250 px
    while y < len(inkrow):
        if inkrow[y] == 0:
            e = y
            while e < len(inkrow) and inkrow[e] == 0: e += 1
            if e - y > 250: keep_rows[y + 120:e - 120] = False
            y = e
        else: y += 1
    c = c[keep_rows]; inkrow = inkrow[keep_rows]
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
    os.makedirs(out, exist_ok=True)
    print("Pass 1/2: tracking scroll position ..."); log = track(a.video, a.fps)
    json.dump({str(k): v for k, v in log.items()}, open(os.path.join(out, "track_log.json"), "w"))
    print("Pass 2/2: compositing ..."); canvas = compose(a.video, a.fps, log)
    cv2.imwrite(os.path.join(out, "full_canvas.png"), canvas)
    pages = paginate(canvas); ims = []
    for i, p in enumerate(pages, 1):
        cv2.imwrite(os.path.join(out, f"page_{i:02d}.png"), p)
        ims.append(Image.fromarray(cv2.cvtColor(p, cv2.COLOR_BGR2RGB)))
    ims[0].save(out + ".pdf", save_all=True, append_images=ims[1:], resolution=150)
    print(f"Done: {len(pages)} pages -> {out}.pdf and {out}/")
