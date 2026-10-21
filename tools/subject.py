import cv2, numpy as np

def subject_mask(img):
    """Foreground mask for a product shot on a plain backdrop.

    The backdrop is modelled as a per-row tone (wall above, table below, seam in
    between) plus a smooth per-column light falloff, fitted only on pixels that are
    not yet considered subject. Colored or clearly lighter/darker pixels are subject;
    soft neutral shadows on the table are treated as backdrop.
    """
    h, w = img.shape[:2]
    f = 8
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    small = cv2.resize(lab, (w // f, h // f), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), 1.2)
    L = small[..., 0] * 100 / 255
    chroma = np.hypot(small[..., 1] - 128, small[..., 2] - 128)
    # backdrop chroma reference (backdrop is near neutral but may have a slight cast)
    fg = chroma > 14
    for _ in range(3):
        Lb = np.where(fg, np.nan, L)
        with np.errstate(all="ignore"):
            rowm = np.nanmedian(Lb, axis=1)
        ok = ~np.isnan(rowm)
        rowm = np.interp(np.arange(len(rowm)), np.nonzero(ok)[0], rowm[ok]) if ok.any() else np.full(len(rowm), np.nanmedian(L))
        res = Lb - rowm[:, None]
        with np.errstate(all="ignore"):
            colc = np.nanmedian(res, axis=0)
        okc = ~np.isnan(colc)
        colc = np.interp(np.arange(len(colc)), np.nonzero(okc)[0], colc[okc]) if okc.any() else np.zeros(len(colc))
        colc = cv2.GaussianBlur(colc.reshape(1, -1).astype(np.float32), (0, 0), 8).ravel()
        bg = rowm[:, None] + colc[None, :]
        # per-row linear light falloff (side lighting on the table), fitted on backdrop only
        xs = np.arange(L.shape[1], dtype=np.float32)
        slope = np.zeros(L.shape[0], np.float32); icpt = np.zeros(L.shape[0], np.float32)
        for yy in range(L.shape[0]):
            okp = ~fg[yy]
            if okp.sum() > 0.15 * L.shape[1] and (xs[okp].max() - xs[okp].min()) > 0.5 * L.shape[1]:
                r = L[yy, okp] - bg[yy, okp]
                A = np.vstack([xs[okp] - xs.mean(), np.ones(okp.sum())]).T
                coef, *_ = np.linalg.lstsq(A, r, rcond=None)
                slope[yy], icpt[yy] = coef
        slope = cv2.GaussianBlur(slope.reshape(-1, 1), (1, 0), 3).ravel()
        icpt = cv2.GaussianBlur(icpt.reshape(-1, 1), (1, 0), 3).ravel()
        bg = bg + icpt[:, None] + slope[:, None] * (xs - xs.mean())[None, :]
        dL = L - bg
        # local texture level of the backdrop per row (table is speckled)
        with np.errstate(all="ignore"):
            tex = np.nanmedian(np.abs(np.where(fg, np.nan, dL)), axis=1)
        tex = np.nan_to_num(tex, nan=2.0)
        thr = np.maximum(6.0, 3.0 * tex)[:, None]
        shadow = (dL < 0) & (chroma < 10) & (dL > -40)
        fg = ((chroma > 14) | (dL > thr) | (dL < -40)) & ~shadow
        # chroma relative to backdrop cast
        fg |= (chroma > 9) & (np.abs(dL) > thr)
    fg = fg.astype(np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    # fill holes (white buttercream inside a colorful outline)
    ff = fg.copy(); hh, ww = ff.shape
    m2 = np.zeros((hh + 2, ww + 2), np.uint8)
    seeds = [(x, y) for x, y in ((0, 0), (ww - 1, 0), (0, hh // 3), (ww - 1, hh // 3)) if ff[y, x] == 0]
    if seeds:
        for sx, sy in seeds:
            if ff[sy, sx] == 0:
                cv2.floodFill(ff, m2, (sx, sy), 2)
        fg[ff == 0] = 1
    n, lab_, st, _ = cv2.connectedComponentsWithStats(fg)
    keep = np.zeros_like(fg)
    if n > 1:
        big = st[1:, cv2.CC_STAT_AREA].max()
        for i in range(1, n):
            if st[i, cv2.CC_STAT_AREA] > 0.02 * big:
                keep[lab_ == i] = 1
    return cv2.resize(keep, (w, h), interpolation=cv2.INTER_NEAREST)
