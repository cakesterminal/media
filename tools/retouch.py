"""Cakes Terminal retouch pipeline.

Steps: even out side-to-side light falloff -> match backdrop color to the house
reference -> remove the seam -> center + frame the subject in a square -> extend the
backdrop where the original frame does not cover the square -> light finishing sharpen.
Nothing is painted on the cake itself: every background edit is masked away from it.
"""
import cv2, numpy as np, json, sys, os

sys.path.insert(0, os.path.dirname(__file__))
from subject import subject_mask

REF = dict(
    wall_L=88.5, wall_a=1.8, wall_b=0.4,     # house backdrop (Lab, L 0-100)
    out_w=1080, out_h=1350,                   # Instagram 4:5 portrait (Guy's choice)
    render_scale=2,                           # compose at 2x, then downsample for crisp detail
    jpeg_quality=92,                          # sRGB baseline JPEG, ~0.4-1 MB: no IG recompression
    top_margin=0.06, body_bottom=0.935,       # top of topper at 6%, cake body ends at 93.5%
    max_subject_width=0.86,
    max_rotation_deg=4.0,
)

# ---------- color helpers ----------
def srgb_to_lin(x):
    x = x / 255.0
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)

def lin_to_srgb(x):
    x = np.clip(x, 0, 1)
    return 255.0 * np.where(x <= 0.0031308, x * 12.92, 1.055 * x ** (1 / 2.4) - 0.055)

def lab_to_bgr_pixel(L, a, b):
    p = np.uint8([[[np.clip(L * 255 / 100, 0, 255), np.clip(a + 128, 0, 255), np.clip(b + 128, 0, 255)]]])
    return cv2.cvtColor(p, cv2.COLOR_LAB2BGR)[0, 0].astype(np.float64)

# ---------- seam ----------
def subject_extent(mask):
    """Horizontal extent of the main subject, ignoring thin bits / small edge objects."""
    h, w = mask.shape
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask)
    if n <= 1:
        return 0, w - 1
    big = st[1:, cv2.CC_STAT_AREA].max()
    keep = np.zeros_like(mask)
    for i in range(1, n):
        x, y, cw, ch, a = st[i]
        touches = x <= 2 or x + cw >= w - 2
        if a >= 0.15 * big or (a >= 0.02 * big and not touches):
            keep[lab == i] = 1
    colsum = keep.sum(0)
    cols = np.nonzero(colsum > 0.15 * colsum.max())[0]
    return int(cols.min()), int(cols.max())

def clean_mask(mask):
    """Drop small objects touching the left/right frame edge (stray props)."""
    h, w = mask.shape
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask)
    if n <= 1:
        return mask
    big = st[1:, cv2.CC_STAT_AREA].max()
    out = mask.copy()
    for i in range(1, n):
        x, y, cw, ch, a = st[i]
        if (x <= 2 or x + cw >= w - 2) and a < 0.15 * big:
            out[lab == i] = 0
    return out

def board_extent(img, mask, x0, x1, y1):
    """Horizontal extent of the cake board. A white board is almost the same tone as the
    table, but its rim draws a clear edge; the plain table produces none. So: columns
    that contain an edge in the band just above/below the cake body's base, taken as the
    contiguous run around the cake."""
    h, w = mask.shape
    L = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32) * 100 / 255
    r0 = max(0, int(y1 - 0.08 * h))
    band = cv2.GaussianBlur(L[r0:], (0, 0), 2)
    gx = cv2.Sobel(band, cv2.CV_32F, 1, 0, ksize=3) / 8
    gy = cv2.Sobel(band, cv2.CV_32F, 0, 1, ksize=3) / 8
    has = (np.hypot(gx, gy) > 2.0).any(0).astype(np.uint8)
    has = cv2.morphologyEx(has.reshape(1, -1), cv2.MORPH_CLOSE, np.ones((1, 41), np.uint8)).ravel()
    cx = int((x0 + x1) / 2)
    if not has[cx]:
        return None
    l = cx
    while l > 0 and has[l - 1]: l -= 1
    r = cx
    while r < w - 1 and has[r + 1]: r += 1
    if l <= 2 or r >= w - 3:          # ran into the frame edge: not a clean board read
        return (int(l), int(r)) if (r - l) < 0.97 * w else None
    return int(l), int(r)

def find_board_bottom(img, mask, x0, x1, y_cake):
    """White/light cake boards blend into the table; find the lowest row under the cake
    that is lighter + smoother than the table beside it."""
    h, w = mask.shape
    L = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32) * 100 / 255
    L = cv2.GaussianBlur(L, (0, 0), 2)
    cw = x1 - x0
    inner = slice(int(x0 + 0.2 * cw), int(x1 - 0.2 * cw))
    outer = np.r_[0:max(int(x0 - 0.05 * w), 1), min(int(x1 + 0.05 * w), w - 1):w]
    last = y_cake
    gap = 0
    for y in range(y_cake, min(h, y_cake + int(0.25 * h))):
        inside = np.median(L[y, inner])
        outside = np.median(L[y, outer]) if len(outer) > 10 else inside - 10
        if inside - outside > 2.0:
            last = y; gap = 0
        else:
            gap += 1
            if gap > int(0.015 * h):
                break
    return last

def find_seam(img, mask, max_deg=4.0):
    """Wall/table seam as a straight line found jointly in the free strips left and
    right of the subject. Returns dict(y0,y1,x0,x1,angle,one_sided) or None."""
    h, w = mask.shape
    L = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32)
    free = cv2.dilate(mask, np.ones((41, 41), np.uint8)) == 0
    xl, xr = subject_extent(mask)
    profs = []
    for c0, c1 in ((0, max(xl - 25, 0)), (min(xr + 25, w), w)):
        if c1 - c0 < 30:
            profs.append(None); continue
        strip = np.where(free[:, c0:c1], L[:, c0:c1], np.nan)
        with np.errstate(all="ignore"):
            prof = np.nanmedian(strip, axis=1)
        good = ~np.isnan(prof)
        if good.sum() < h * 0.5:
            profs.append(None); continue
        prof = np.interp(np.arange(h), np.nonzero(good)[0], prof[good])
        prof = cv2.GaussianBlur(prof.reshape(-1, 1).astype(np.float32), (1, 0), 5).ravel()
        g = np.abs(np.gradient(prof)); g[: int(0.25 * h)] = 0; g[int(0.8 * h):] = 0
        profs.append(((c0 + c1) / 2, g))
    sides = [p for p in profs if p is not None]
    if not sides:
        return None
    if len(sides) == 1:
        x, g = sides[0]
        y = int(np.argmax(g))
        if g[y] < 0.25:
            return None
        return dict(y0=y, y1=y, x0=0, x1=w, angle=0.0, one_sided=True, strength=float(g[y]))
    (xL, gL), (xR, gR) = sides
    dx = xR - xL
    maxdy = int(np.tan(np.radians(max_deg)) * dx)
    best = (-1, 0, 0)
    for dy in range(-maxdy, maxdy + 1):
        sh = np.roll(gR, -dy)
        sc = gL + sh
        y = int(np.argmax(sc))
        if sc[y] > best[0]:
            best = (float(sc[y]), y, y + dy)
    sc, yl, yr = best
    if sc < 0.5:
        return None
    ang = float(np.degrees(np.arctan2(yr - yl, dx)))
    return dict(y0=yl, y1=yr, x0=xL, x1=xR, angle=ang, one_sided=False, strength=sc)

def seam_y_at(seam, x):
    t = (x - seam["x0"]) / max(seam["x1"] - seam["x0"], 1)
    return seam["y0"] + t * (seam["y1"] - seam["y0"])

def remove_seam(img, mask, seam, band=None):
    """Replace the hard wall/table line with a smooth gradient, background only."""
    h, w = mask.shape
    band = band or int(0.06 * h)
    f = img.astype(np.float32)
    # low-frequency plate: strong vertical blur of the background only
    bgw = (cv2.dilate(mask, np.ones((31, 31), np.uint8)) == 0).astype(np.float32)
    k = (0, 0)
    num = cv2.GaussianBlur(f * bgw[..., None], k, sigmaX=25, sigmaY=band / 2.0)
    den = cv2.GaussianBlur(bgw, k, sigmaX=25, sigmaY=band / 2.0)[..., None] + 1e-4
    plate = num / den
    # detail layer (texture) - attenuated right at the seam line so the crisp edge disappears
    low = cv2.GaussianBlur(f, (0, 0), 4)
    detail = f - low
    lowv = cv2.GaussianBlur(f * bgw[..., None], (0, 0), sigmaX=4, sigmaY=4) / (cv2.GaussianBlur(bgw, (0, 0), 4)[..., None] + 1e-4)
    yy = np.arange(h)[:, None].astype(np.float32)
    xx = np.arange(w)[None, :].astype(np.float32)
    sy = seam_y_at(seam, xx)
    dist = np.abs(yy - sy)
    wband = np.clip(1 - dist / band, 0, 1) ** 1.5           # 1 at seam, 0 at band edge
    wline = np.clip(1 - dist / 22.0, 0, 1) ** 0.7             # kill detail around the line
    newlow = plate * wband[..., None] + lowv * (1 - wband[..., None])
    out = newlow + detail * (1 - 0.95 * wline[..., None])
    # protect the cake: feathered mask
    prot = cv2.GaussianBlur(cv2.dilate(mask, np.ones((15, 15), np.uint8)).astype(np.float32), (0, 0), 6)
    prot = np.clip(prot * 1.5, 0, 1)[..., None]
    res = f * prot + out * (1 - prot)
    # only touch the band
    m = (wband > 0)[..., None]
    return np.where(m, res, f).clip(0, 255).astype(np.uint8)

# ---------- background extension ----------
def extend_background(canvas, valid, mask_c):
    """Fill uncovered canvas pixels by mirroring the nearest backdrop texture, then
    matching its low-frequency tone to the neighbouring backdrop and feathering."""
    h, w = valid.shape
    out = canvas.astype(np.float32).copy()
    filled = valid.copy()
    bg_ok = valid & ~(cv2.dilate(mask_c, np.ones((25, 25), np.uint8)) > 0)
    # horizontal pass, row by row
    for y in range(h):
        row_v = np.nonzero(valid[y])[0]
        if len(row_v) == 0:
            continue
        a, b = row_v.min(), row_v.max()
        for side, gap in (("L", range(0, a)), ("R", range(b + 1, w))):
            gap = list(gap)
            if not gap:
                continue
            # usable backdrop texture: nearest run of clean backdrop (>= 40 px) in this row
            ok = bg_ok[y].astype(np.int8)
            edges = np.diff(np.r_[0, ok, 0])
            starts, ends = np.nonzero(edges == 1)[0], np.nonzero(edges == -1)[0]
            runs = [(st_, en) for st_, en in zip(starts, ends) if en - st_ >= 40]
            if not runs:
                runs = [(st_, en) for st_, en in zip(starts, ends)] or [(a, b + 1)]
                runs = [max(runs, key=lambda r: r[1] - r[0])]
            ref_x = a if side == "L" else b
            st_, en = min(runs, key=lambda r: min(abs(r[0] - ref_x), abs(r[1] - ref_x)))
            strip = out[y, st_:en] if side == "L" else out[y, st_:en][::-1]
            n = len(strip)
            for i, x in enumerate(gap if side == "R" else gap[::-1]):
                d = i  # distance from edge
                period = 2 * n
                k = d % period
                idx = k if k < n else period - 1 - k
                out[y, x] = strip[idx]
            filled[y, gap] = True
    # vertical pass for rows with nothing (top/bottom)
    rows = np.nonzero(filled.any(1))[0]
    if len(rows):
        t, btm = rows.min(), rows.max()
        n = max(1, min(200, btm - t))
        for y in range(0, t):
            d = t - 1 - y; k = d % (2 * n); idx = k if k < n else 2 * n - 1 - k
            out[y] = out[t + idx]
        for y in range(btm + 1, h):
            d = y - btm - 1; k = d % (2 * n); idx = k if k < n else 2 * n - 1 - k
            out[y] = out[btm - idx]
    # tone-match the filled area to a smooth plate built from real backdrop
    filled_area = ~valid
    if filled_area.any():
        # low-res backdrop plate: average real backdrop per cell, inpaint empty cells
        f = 16
        hs, ws = h // f, w // f
        bw = bg_ok.astype(np.float32)
        num = cv2.resize(out * bw[..., None], (ws, hs), interpolation=cv2.INTER_AREA)
        den = cv2.resize(bw, (ws, hs), interpolation=cv2.INTER_AREA)
        cell = num / np.maximum(den[..., None], 1e-6)
        holes = (den < 0.2).astype(np.uint8)
        cell8 = np.clip(cell, 0, 255).astype(np.uint8)
        cell8 = cv2.inpaint(cell8, holes, 6, cv2.INPAINT_TELEA)
        plate = cv2.resize(cell8.astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
        plate = cv2.GaussianBlur(plate, (0, 0), 20)
        lowf = cv2.GaussianBlur(out, (0, 0), 25)
        corr = (plate - lowf)
        out = np.where(filled_area[..., None], out + corr, out)
        # feather the boundary
        fa = cv2.GaussianBlur(filled_area.astype(np.float32), (0, 0), 8)
        seam_zone = (fa > 0.02) & (fa < 0.98)
        sm = cv2.GaussianBlur(out, (0, 0), 3)
        out = np.where(seam_zone[..., None] & ~(mask_c[..., None] > 0), 0.5 * out + 0.5 * sm, out)
    return out.clip(0, 255).astype(np.uint8)

# ---------- uneven side lighting ----------
def even_light(img, mask, max_gain=1.35):
    """Correct a left-to-right (or right-to-left) light falloff. The falloff is measured
    on the backdrop only (each row normalised to its own level, so wall/table/seam don't
    matter), fitted with a smooth quadratic across the width, and divided out of the
    whole image in linear light - the cake gets the same correction as the backdrop it
    stands in front of, so the scene reads as evenly lit."""
    h, w = mask.shape
    lin = srgb_to_lin(img.astype(np.float64))
    Y = lin @ np.array([0.0722, 0.7152, 0.2126])          # BGR luminance
    free = cv2.dilate(mask, np.ones((61, 61), np.uint8)) == 0
    f = 8
    Ys = cv2.resize(Y, (w // f, h // f), interpolation=cv2.INTER_AREA)
    Fs = cv2.resize(free.astype(np.uint8), (w // f, h // f), interpolation=cv2.INTER_NEAREST) > 0
    Yb = np.where(Fs, Ys, np.nan)
    with np.errstate(all="ignore"):
        rowref = np.nanmedian(Yb, axis=1, keepdims=True)
        rel = Yb / rowref
        prof = np.nanmedian(rel, axis=0)
        cnt = np.sum(Fs, axis=0)
    xs = (np.arange(len(prof)) + 0.5) / len(prof)
    ok = ~np.isnan(prof) & (cnt > 0.1 * Fs.shape[0])
    if ok.sum() < 0.3 * len(prof) or (xs[ok].max() - xs[ok].min()) < 0.6:
        return img, None                                   # not enough backdrop on both sides
    coef = np.polyfit(xs[ok], prof[ok], 2, w=np.sqrt(cnt[ok]))
    xf = (np.arange(w) + 0.5) / w
    fit = np.polyval(coef, xf)
    fit = fit / np.mean(np.polyval(coef, xs[ok]))          # keep overall exposure unchanged
    spread = float(fit.max() / fit.min() - 1)
    if spread < 0.04:                                      # under ~4% edge-to-edge: leave it
        return img, round(spread * 100, 1)
    gain = np.clip(1 / fit, 1 / max_gain, max_gain)
    out = lin * gain[None, :, None]
    out = np.where(out > 0.9, 0.9 + (1 - np.exp(-(out - 0.9) / 0.1)) * 0.1, out)
    return lin_to_srgb(out).astype(np.uint8), round(spread * 100, 1)

# ---------- delivery ----------
def save_for_social(bgr, path_out, ref=REF):
    """Baseline sRGB JPEG, 4:2:0, quality tuned so the file lands well under Instagram's
    re-encode thresholds (1080 px wide, < ~1.6 MB)."""
    from PIL import Image
    im = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    q = ref["jpeg_quality"]
    while True:
        im.save(path_out, "JPEG", quality=q, optimize=True, progressive=False, subsampling=2)
        if os.path.getsize(path_out) < 1_500_000 or q <= 78:
            break
        q -= 3
    return q

def prepare_ready(path_in, path_out, ref=REF):
    """Photos Guy already retouched: no pixel edits. Fit to 4:5 portrait (center crop of
    the excess, flagged) and size/compress for upload at 1080x1350."""
    img = cv2.imread(path_in)
    h, w = img.shape[:2]
    note = None
    r = w / h
    target = ref["out_w"] / ref["out_h"]                 # 0.8
    if r < target - 0.005:                          # taller than 4:5: trim top/bottom
        nh = int(round(w / target)); y0 = (h - nh) // 2
        img = img[y0:y0 + nh]; note = f"trimmed top/bottom to 4:5 (was {w}x{h})"
    elif r > target + 0.005:                        # wider than 4:5: trim sides
        nw = int(round(h * target)); x0 = (w - nw) // 2
        img = img[:, x0:x0 + nw]; note = f"trimmed sides to 4:5 (was {w}x{h})"
    h, w = img.shape[:2]
    if w != ref["out_w"]:
        nh = int(round(h * ref["out_w"] / w))
        img = cv2.resize(img, (ref["out_w"], nh), interpolation=cv2.INTER_AREA if w > ref["out_w"] else cv2.INTER_CUBIC)
    q = save_for_social(img, path_out, ref)
    return dict(size=img.shape[1::-1], quality=q, note=note, upscaled=w < ref["out_w"])

# ---------- main ----------
def retouch(path_in, path_out, ref=REF, debug=None):
    img = cv2.imread(path_in)
    h, w = img.shape[:2]
    log = {}
    raw0 = subject_mask(img)
    mask = clean_mask(raw0)
    stray0 = ((raw0 > 0) & (mask == 0)).astype(np.uint8)   # props cut off at the frame edge

    # 1. level the horizon using the seam
    seam = find_seam(img, mask)
    log["seam"] = seam
    # Auto-leveling disabled: seam/edge tilt proved unreliable against Guy's own edits.
    # Straightening is left to Lightroom before export.
    if False and seam and not seam["one_sided"] and abs(seam["angle"]) < ref["max_rotation_deg"] and abs(seam["angle"]) > 0.3:
        R = cv2.getRotationMatrix2D((w / 2, h / 2), seam["angle"], 1.0)
        img = cv2.warpAffine(img, R, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
        log["rotated_deg"] = round(seam["angle"], 2)
        mask = clean_mask(subject_mask(img))
        seam = find_seam(img, mask)
        log["seam_after_level"] = seam

    # even out side-to-side light falloff before anything is measured for color
    img, falloff = even_light(img, mask)
    log["side_falloff_pct"] = falloff

    xl, xr = subject_extent(mask)
    ys_, xs_ = np.nonzero(mask)
    edges_touched = 0
    if len(ys_):
        edges_touched = sum([xs_.min() < 0.03 * w, xs_.max() > 0.97 * w, ys_.min() < 0.03 * h, ys_.max() > 0.97 * h])
    # flat lay / close-up: colored product spread over (almost) the entire frame
    labf = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    colored = np.hypot(labf[..., 1] - 128, labf[..., 2] - 128) > 14
    rr = np.nonzero(colored.mean(1) > 0.02)[0]; cc = np.nonzero(colored.mean(0) > 0.02)[0]
    spread = len(rr) and len(cc) and (rr.max() - rr.min()) > 0.9 * h and (cc.max() - cc.min()) > 0.9 * w
    full = mask.mean() > 0.5 or (xl < 0.03 * w and xr > 0.97 * w) or ((xl < 0.03 * w or xr > 0.97 * w) and (xr - xl) > 0.85 * w) or edges_touched >= 3 or bool(spread)
    log["mode"] = "full-frame" if full else "standing cake"

    # 2. color: match backdrop wall to the house reference with white-balance style gains
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    free = cv2.dilate(mask, np.ones((61, 61), np.uint8)) == 0
    wall = np.zeros_like(free)
    sy = int(seam_y_at(seam, w / 2)) if seam else int(0.55 * h)
    wall[int(0.03 * h):max(int(0.03 * h) + 10, sy - int(0.04 * h))] = True
    wall &= free
    if wall.sum() < 3000:     # flat lays / no wall visible: use whatever backdrop exists
        wall = free
    if wall.sum() > 3000:
        cur = np.median(img[wall].astype(np.float64), axis=0)            # BGR
        tgt = lab_to_bgr_pixel(ref["wall_L"], ref["wall_a"], ref["wall_b"])
        if full:
            # no wall in shot: neutralise the color cast only, keep exposure within +/-3 L
            cur_lab0 = cv2.cvtColor(cur.reshape(1, 1, 3).astype(np.uint8), cv2.COLOR_BGR2LAB)[0, 0].astype(float)
            Lnow = cur_lab0[0] * 100 / 255
            tgt = lab_to_bgr_pixel(Lnow + np.clip(ref["wall_L"] - Lnow, -3, 3), ref["wall_a"], ref["wall_b"])
        gains = srgb_to_lin(tgt) / np.maximum(srgb_to_lin(cur), 1e-4)
        gains = np.clip(gains, 0.6, 1.8)
        lin = srgb_to_lin(img.astype(np.float64)) * gains
        # soft highlight roll-off so whites on the cake don't clip
        lin = np.where(lin > 0.85, 0.85 + (1 - np.exp(-(lin - 0.85) / 0.15)) * 0.15, lin)
        img = lin_to_srgb(lin).astype(np.uint8)
        cur_lab = cv2.cvtColor(cur.reshape(1, 1, 3).astype(np.uint8), cv2.COLOR_BGR2LAB)[0, 0].astype(float)
        log["wall_before_Lab"] = [round(cur_lab[0] * 100 / 255, 1), round(cur_lab[1] - 128, 1), round(cur_lab[2] - 128, 1)]
        log["gains_BGR"] = [round(float(g), 3) for g in gains]

    # 3. seam removal (background only)
    if seam:
        img = remove_seam(img, mask, seam)
        log["seam_removed"] = True

    # 4. framing
    W = ref["out_w"] * ref["render_scale"]; H = ref["out_h"] * ref["render_scale"]
    ys, xs = np.nonzero(mask)
    if len(ys) == 0 or full:
        # flat lay / close-up filling the frame: cover the 3:4 canvas, keep the product's base in view
        s = max(W / w, H / h)
        tx = (W - w * s) / 2
        crop_h = H / s                                    # source rows that fit the canvas
        labc = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
        chroma = np.hypot(labc[..., 1] - 128, labc[..., 2] - 128)
        crow = (chroma > 14).mean(1)
        crow = cv2.GaussianBlur(crow.reshape(-1, 1).astype(np.float32), (1, 0), 5).ravel()
        rows = np.nonzero(crow > 0.05)[0]
        if len(rows) == 0 or spread or (rows.min() < 0.05 * h and rows.max() > 0.95 * h):
            top = (h - crop_h) / 2                        # flat lay: center crop
        else:
            bottom = min(h, rows.max() + 0.03 * h)
            top = float(np.clip(bottom - crop_h, 0, max(h - crop_h, 0)))
        ty = -top * s
        log["framing"] = dict(type="full-frame", top=int(top), scale=round(s, 3))
    else:
        rowcov = mask.sum(1)
        maxw = rowcov.max()
        rows = np.nonzero(rowcov > 0.25 * maxw)[0]
        y0, y1 = ys.min(), rows.max()
        x0, x1 = xl, xr
        # cake body (without a white board) ends ~93.5% down in Guy's frames
        target_h = H * (ref["body_bottom"] - ref["top_margin"])
        s = target_h / max(y1 - y0, 1)
        be = board_extent(img, mask, x0, x1, y1)
        wx0, wx1 = (min(x0, be[0]), max(x1, be[1])) if be else (x0, x1)
        log["board_x"] = be
        if (wx1 - wx0) * s > ref["max_subject_width"] * W:
            s = ref["max_subject_width"] * W / (wx1 - wx0)
        # keep the whole thing (board included) inside the frame when centering on the cake
        cx_body = (x0 + x1) / 2
        half = W / 2 / s
        cx_lo, cx_hi = wx1 - half + 0.02 * W / s, wx0 + half - 0.02 * W / s
        x0, x1 = (cx_body - (x1 - x0) / 2, cx_body + (x1 - x0) / 2) if cx_lo > cx_hi else (
            np.clip(cx_body, cx_lo, cx_hi) - (x1 - x0) / 2, np.clip(cx_body, cx_lo, cx_hi) + (x1 - x0) / 2)
        cx = (x0 + x1) / 2
        tx = W / 2 - cx * s
        ty = H * ref["body_bottom"] - y1 * s
        log["framing"] = dict(bbox=[int(x0), int(y0), int(x1), int(y1)], scale=round(s, 3))
    M = np.float32([[s, 0, tx], [0, s, ty]])
    interp = cv2.INTER_CUBIC if s > 1 else cv2.INTER_AREA
    canvas = cv2.warpAffine(img, M, (W, H), flags=interp, borderValue=(0, 0, 0))
    valid = cv2.warpAffine(np.full((h, w), 255, np.uint8), M, (W, H), flags=cv2.INTER_NEAREST) > 0
    valid = cv2.erode(valid.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    mask_c = cv2.warpAffine(mask, M, (W, H), flags=cv2.INTER_NEAREST)
    if not full:
        # stray props cut off at the frame edge: treat as missing and rebuild as backdrop
        stray = cv2.dilate(stray0, np.ones((31, 31), np.uint8))
        stray_c = cv2.warpAffine(stray, M, (W, H), flags=cv2.INTER_NEAREST) > 0
        valid &= ~stray_c
        # holes outside the subject: invalidate out to the frame edge so the fill is clean
        cx_l = int(xl * s + tx)
        cx_r = int(xr * s + tx)
        for y in np.nonzero((~valid).any(1))[0]:
            bad = np.nonzero(~valid[y])[0]
            r = bad[bad > cx_r]
            if len(r): valid[y, r.min():] = False
            l = bad[bad < cx_l]
            if len(l): valid[y, :l.max() + 1] = False
    log["uncovered_pct"] = round(100 * (1 - valid.mean()), 1)
    if (~valid).any():
        canvas = extend_background(canvas, valid, mask_c)
    if s / ref["render_scale"] > 1.6:
        log["warning"] = f"upscaled {s / ref['render_scale']:.1f}x - export a larger file for best sharpness"

    # 5. finishing: downsample to the delivery size, then a light output sharpen
    canvas = cv2.resize(canvas, (ref["out_w"], ref["out_h"]), interpolation=cv2.INTER_AREA)
    valid = cv2.resize(valid.astype(np.uint8), (ref["out_w"], ref["out_h"]), interpolation=cv2.INTER_NEAREST) > 0
    blur = cv2.GaussianBlur(canvas, (0, 0), 0.8)
    canvas = cv2.addWeighted(canvas, 1.25, blur, -0.25, 0)
    save_for_social(canvas, path_out, ref)
    if debug:
        dbg = canvas.copy()
        dbg[~valid] = (0.6 * dbg[~valid] + [0, 80, 0]).astype(np.uint8)
        cv2.imwrite(debug, cv2.resize(dbg, (600, 800)))
    return log

if __name__ == "__main__":
    for p in sys.argv[1:]:
        base = os.path.splitext(os.path.basename(p))[0]
        log = retouch(p, f"out/{base}_final.jpg", debug=f"work/{base}_dbg.jpg")
        print(base, json.dumps(log, default=str))
