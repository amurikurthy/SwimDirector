"""
Automatic pool geometry from the overview camera.

1. Background frame = per-pixel median of frames over a few seconds (swimmers vanish).
2. Water = the large blue/teal region. Lane ropes = thin non-water lines inside it.
3. Lens: the distortion that makes every rope straight (division model, see overview.py).
4. Ropes (undistorted) -> lines; wall lines through where the ropes end.
5. Rope/wall intersections + known lane spacing -> homography -> 4 pool corners.

Conventions (flippable from the admin page): lane 1 is the side nearest the overview
camera (bottom of the image); the start wall is the left end of the ropes in the image.
"""
import math

import cv2
import numpy as np

from .overview import fit_lens, lens_distort, lens_undistort


class GeometryError(Exception):
    pass


def background(frames):
    """Median of a list of BGR frames (same size)."""
    stack = np.stack(frames[-25:], 0)
    return np.median(stack, axis=0).astype(np.uint8)


def _kern(n):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (n, n))


def rope_mask(img):
    """Returns (rope_mask, water_hull) at the image's own resolution."""
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    water = cv2.inRange(hsv, (75, 45, 60), (112, 255, 255))
    water = cv2.morphologyEx(water, cv2.MORPH_OPEN, _kern(3))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(cv2.morphologyEx(
        water, cv2.MORPH_CLOSE, _kern(15)))
    if n < 2:
        raise GeometryError("no water found in the overview")
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if stats[i, cv2.CC_STAT_AREA] < 0.1 * w * h:
        raise GeometryError("water region too small - is the overview pointed at the pool?")
    region = (lab == i).astype(np.uint8) * 255
    pts = cv2.findNonZero(region)
    hull = np.zeros_like(region)
    cv2.fillConvexPoly(hull, cv2.convexHull(pts), 255)
    inside = cv2.erode(hull, _kern(max(5, w // 150)))
    b, g, r = [img[..., i].astype(np.int16) for i in range(3)]
    v, s = hsv[..., 2], hsv[..., 1]
    green = (g > b + 8) & (g > r + 15) & (v < 170)          # green rope discs
    white = (s < 50) & (v > 185)                              # white / pale rope discs
    gold = (r > b + 30) & (g > b + 20) & (v > 90)             # yellow / gold ropes
    red = (r > g + 50) & (r > b + 50)                         # red marker discs
    cand = ((green | white | gold | red).astype(np.uint8) * 255) | cv2.bitwise_not(water)
    cand = cv2.bitwise_and(cand, inside)
    # keep thin structures: ropes are a few px wide, people/flags/deck are blobs
    k = max(7, (w // 90) | 1)
    thick = cv2.morphologyEx(cand, cv2.MORPH_OPEN, _kern(k))
    thin = cv2.subtract(cand, cv2.dilate(thick, _kern(5)))
    thin = cv2.morphologyEx(thin, cv2.MORPH_CLOSE, _kern(3))
    return thin, hull, region


def _line_kernel(length, ang_deg):
    k = np.zeros((length, length), np.uint8)
    c = length // 2
    dx, dy = math.cos(math.radians(ang_deg)) * c, math.sin(math.radians(ang_deg)) * c
    cv2.line(k, (int(round(c - dx)), int(round(c - dy))), (int(round(c + dx)), int(round(c + dy))), 1, 1)
    return k


def directional(mask, w):
    """Keep only line-like structure along the dominant (rope) direction; cuts ropes free
    from walls, flag lines and people that touch them."""
    segs = cv2.HoughLinesP(mask, 1, np.pi / 180, threshold=30, minLineLength=max(20, w // 50),
                           maxLineGap=4)
    if segs is None:
        raise GeometryError("no line structure in the pool")
    x1, y1, x2, y2 = segs.reshape(-1, 4).T.astype(np.float64)
    ang = np.degrees(np.arctan2(y2 - y1, x2 - x1))
    ang = (ang + 90) % 180 - 90
    ln = np.hypot(x2 - x1, y2 - y1)
    keep = np.abs(ang) < 60
    if not keep.any():
        raise GeometryError("no rope-like lines found")
    a, l = ang[keep], ln[keep]
    o = np.argsort(a)
    cum = np.cumsum(l[o])
    dom = float(a[o][np.searchsorted(cum, cum[-1] / 2)])
    L = max(15, (w // 60) | 1)
    out = np.zeros_like(mask)
    for d in np.arange(dom - 24, dom + 24.1, 4):
        out = cv2.max(out, cv2.morphologyEx(mask, cv2.MORPH_OPEN, _line_kernel(L, d)))
    return out, dom


def _components(mask, min_px=60):
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = []
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < min_px:
            continue
        ys, xs = np.nonzero(lab == i)
        P = np.stack([xs, ys], 1).astype(np.float64)
        if len(P) > 2000:
            P = P[np.random.default_rng(0).choice(len(P), 2000, replace=False)]
        out.append(P)
    return out


def _line(P):
    m = P.mean(0)
    _, _, vt = np.linalg.svd(P - m, full_matrices=False)
    d = vt[0]
    if d[0] < 0:
        d = -d
    n = np.array([-d[1], d[0]])
    return m, d, n


def chain_ropes(mask, dom, w, h):
    """Group rope fragments into whole ropes in the (distorted) image.
    Works in coordinates rotated so ropes run roughly along +u; each rope is a
    quadratic v(u), so lens curvature is fine."""
    th = math.radians(dom)
    c = np.array([w / 2, h / 2])
    Rm = np.array([[math.cos(th), math.sin(th)], [-math.sin(th), math.cos(th)]])

    def rot(P):
        return (P - c) @ Rm.T

    comps = []
    for P in _components(mask):
        U = rot(P)
        span = np.ptp(U[:, 0])
        thick = np.ptp(np.percentile(U[:, 1], [10, 90]))
        if span < 0.03 * w or span < 3 * max(thick, 1):
            continue
        comps.append({"P": P, "U": U, "span": span})
    comps.sort(key=lambda d: -d["span"])

    def fit(U):
        deg = 2 if np.ptp(U[:, 0]) > 0.2 * w else 1
        return np.polyfit(U[:, 0], U[:, 1], deg)

    ropes = []
    for cpt in comps:
        best, best_r = None, None
        for r in ropes:
            u0, u1 = r["U"][:, 0].min(), r["U"][:, 0].max()
            cu = cpt["U"][:, 0]
            overlap = np.mean((cu > u0) & (cu < u1))
            if overlap > 0.5 and cpt["span"] > 0.15 * w:
                continue                       # side by side, not a continuation
            gap = max(0.0, cu.min() - u1, u0 - cu.max())
            res = np.median(np.abs(np.polyval(r["poly"], cu) - cpt["U"][:, 1]))
            tol = 0.005 * h + 0.012 * gap
            if res < tol and (best is None or res < best):
                best, best_r = res, r
        if best_r is None:
            ropes.append({"U": cpt["U"], "P": cpt["P"], "poly": fit(cpt["U"])})
        else:
            best_r["U"] = np.concatenate([best_r["U"], cpt["U"]])
            best_r["P"] = np.concatenate([best_r["P"], cpt["P"]])
            best_r["poly"] = fit(best_r["U"])
    for r in ropes:
        r["span"] = float(np.ptp(r["U"][:, 0]))
    return ropes


def _undistort_canvas(mask, k, size, scale=0.5):
    """Remap a (distorted) mask into undistorted space on a canvas big enough for the
    stretched corners. Returns (mask_u, to_canvas(pts_u), from_canvas(pts_c))."""
    w, h = size
    border = np.float64([[0, 0], [w, 0], [w, h], [0, h], [w / 2, 0], [w / 2, h], [0, h / 2], [w, h / 2]])
    bu = lens_undistort(border, k, size)
    x0, y0 = bu.min(0)
    x1, y1 = bu.max(0)
    # cap runaway stretch at the very corners
    x0, y0 = max(x0, -0.6 * w), max(y0, -0.6 * h)
    x1, y1 = min(x1, 1.6 * w), min(y1, 1.6 * h)
    cw, ch = int((x1 - x0) * scale), int((y1 - y0) * scale)
    gx, gy = np.meshgrid(np.arange(cw, dtype=np.float64), np.arange(ch, dtype=np.float64))
    pu = np.stack([gx.ravel() / scale + x0, gy.ravel() / scale + y0], 1)
    pd = lens_distort(pu, k, size).astype(np.float32)
    mapx = pd[:, 0].reshape(ch, cw)
    mapy = pd[:, 1].reshape(ch, cw)
    mu = cv2.remap(mask, mapx, mapy, cv2.INTER_NEAREST, borderValue=0)

    def to_c(P):
        return (np.asarray(P, np.float64) - [x0, y0]) * scale

    def from_c(P):
        return np.asarray(P, np.float64) / scale + [x0, y0]
    return mu, to_c, from_c


def _hough_ropes(mu, dom_deg):
    """Straight lines in the undistorted rope mask, along the rope direction.
    Returns list of (point, direction, normal, votes) in canvas coords."""
    ch, cw = mu.shape
    thr = max(40, int(0.08 * cw))
    L = cv2.HoughLines(mu, 1, np.pi / 360, thr)
    if L is None:
        return []
    L = L.reshape(-1, L.shape[-1])
    out = []
    for row in L[:400]:
        rho, theta = float(row[0]), float(row[1])
        d = np.array([-math.sin(theta), math.cos(theta)])
        if d[0] < 0:
            d = -d
        ang = math.degrees(math.atan2(d[1], d[0]))
        if abs(((ang - dom_deg) + 90) % 180 - 90) > 30:
            continue
        n = np.array([math.cos(theta), math.sin(theta)])
        p0 = n * rho
        dup = False
        for q in out:
            # same rope if close along the whole visible width
            xs = np.array([0.0, cw / 2, cw])
            ya = [(p0 + d * ((x - p0[0]) / d[0]))[1] for x in xs]
            yb = [(q[0] + q[1] * ((x - q[0][0]) / q[1][0]))[1] for x in xs]
            if max(abs(a - b) for a, b in zip(ya, yb)) < 0.012 * ch + 4:
                dup = True
                break
        if not dup:
            out.append((p0, d, n, len(out)))
    return out


def detect(img, lanes, lane_width_m, length_m, outer_ropes="auto",
           flip_lanes=False, flip_ends=False):
    """Returns dict(corners=[[x,y]*4 normalized], lens={'k':k}, ropes=n, report=str,
    quality=0..1, debug=...). Raises GeometryError with a readable reason."""
    import itertools
    h, w = img.shape[:2]
    size = (w, h)
    mask, hull, water = rope_mask(img)
    mask, dom = directional(mask, w)

    # 1) lens from the longest clean rope pieces
    chains = chain_ropes(mask, dom, w, h)
    chains = sorted([c for c in chains if c["span"] > 0.25 * w], key=lambda c: -c["span"])[:5]
    if chains:
        k, before, after = fit_lens([c["P"] for c in chains], size)
    else:
        k, before, after = 0.0, 0.0, 0.0

    # 2) rope lines in undistorted space: fragments of the same rope are collinear there
    frags = []
    for P in _components(mask, min_px=25):
        U = lens_undistort(P, k, size)
        m_, d_, n_ = _line(U)
        t = U @ d_
        span = float(np.ptp(t))
        if span < 0.015 * w or np.ptp(np.percentile(U @ n_, [10, 90])) > 0.25 * span:
            continue
        ang = math.degrees(math.atan2(d_[1], d_[0]))
        if abs(((ang - dom) + 90) % 180 - 90) > 30:
            continue
        frags.append({"U": U, "m": m_, "d": d_, "n": n_, "span": span})
    frags.sort(key=lambda f: -f["span"])
    ropes = []
    tol = 0.006 * h + 2
    for f in frags:
        best = None
        for r in ropes:
            dist = np.median(np.abs((f["U"] - r["m"]) @ r["n"]))
            if f["span"] > 0.05 * w and abs(float(f["d"] @ r["n"])) > math.sin(math.radians(4)):
                continue
            if dist < tol and (best is None or dist < best[0]):
                best = (dist, r)
        if best is None:
            ropes.append(dict(f))
        else:
            r = best[1]
            r["U"] = np.concatenate([r["U"], f["U"]])
            r["m"], r["d"], r["n"] = _line(r["U"])
            r["span"] = float(np.ptp(r["U"] @ r["d"]))
    if not ropes:
        raise GeometryError("could not find lane ropes")
    maxspan = max(r["span"] for r in ropes)
    ropes = [r for r in ropes if r["span"] > 0.3 * maxspan]
    lines = [(r["m"], r["d"], r["n"], r) for r in ropes]
    # keep lines consistent with one vanishing point (ropes are parallel in the pool)
    if len(lines) >= 3:
        best = None
        for (pa, da, _, _), (pb, db, _, _) in itertools.combinations(lines, 2):
            A = np.array([da, -db]).T
            if abs(np.linalg.det(A)) < 1e-9:
                continue
            t = np.linalg.solve(A, pb - pa)
            vp = pa + t[0] * da
            inl = []
            for (p, d, n, _) in lines:
                v = vp - p
                ang = abs(math.degrees(math.atan2(d[0] * v[1] - d[1] * v[0], d @ v)))
                inl.append(min(ang, 180 - ang) < 1.2)
            cnt = sum(inl)
            if best is None or cnt > best[0]:
                best = (cnt, inl)
        if best and best[0] >= 2:
            lines = [l for l, ok in zip(lines, best[1]) if ok]

    # work in undistorted image coords (canvas = identity) from here on
    to_c = from_c = lambda P: np.asarray(P, np.float64)
    md = cv2.dilate(mask, _kern(3))
    mu, _, _ = None, None, None
    # distance transform in undistorted space, sampled lazily through the lens
    dt_raw = np.minimum(cv2.distanceTransform(cv2.bitwise_not(md), cv2.DIST_L2, 3), 12.0).astype(np.float32)
    ch, cw = h, w

    def support(P):
        Pd = lens_distort(np.asarray(P, np.float64), k, size).astype(np.float32)
        ok = (Pd[:, 0] >= 0) & (Pd[:, 0] < w - 1) & (Pd[:, 1] >= 0) & (Pd[:, 1] < h - 1)
        d = np.full(len(Pd), 12.0, np.float32)
        if ok.any():
            d[ok] = cv2.remap(dt_raw, Pd[ok, 0].reshape(1, -1), Pd[ok, 1].reshape(1, -1),
                              cv2.INTER_LINEAR).ravel()
        return np.exp(-d ** 2 / (2 * 2.0 ** 2)), ok

    # 3) rope ends -> wall lines
    ends_a, ends_b = [], []
    for p, d, n, r in lines:
        t = np.sort(r["U"] @ d)
        gaps = np.diff(t)
        lo, hi = np.percentile(t, 0.5), np.percentile(t, 99.5)
        base = p - (p @ d) * d
        ends_a.append(base + lo * d)
        ends_b.append(base + hi * d)

    def wall_candidates(E, n_best=6):
        """Distinct candidate wall lines through the rope ends (best supported first)."""
        E = np.array([e for e in E if e is not None])
        if len(E) < 2:
            raise GeometryError("not enough rope ends to find the walls")
        tol = 0.02 * h
        cands = []
        for i in range(len(E)):
            for j in range(i + 1, len(E)):
                dd = E[j] - E[i]
                if np.linalg.norm(dd) < 1e-6:
                    continue
                nn = np.array([-dd[1], dd[0]]) / np.linalg.norm(dd)
                inl = np.abs((E - E[i]) @ nn) < tol
                m_, d_, n_ = _line(E[inl]) if inl.sum() >= 2 else _line(E[[i, j]])
                cands.append((int(inl.sum()), m_, n_))
        cands.sort(key=lambda c: -c[0])
        out = []
        for c in cands:
            if all(abs(float(c[2] @ o[2])) < 0.999 or abs(float((c[1] - o[1]) @ o[2])) > tol for o in out):
                out.append(c)
            if len(out) >= n_best:
                break
        return [(m_, n_) for _, m_, n_ in out]

    def inter(m1, d1, m2, n2):
        t = float((m2 - m1) @ n2) / float(d1 @ n2)
        return m1 + t * d1

    walls_a = wall_candidates(ends_a)
    walls_b = wall_candidates(ends_b)
    ma, na = walls_a[0]
    mb, nb = walls_b[0]

    # order across the pool, nearest the camera (bottom) first
    mean_d = np.mean([l[1] for l in lines], 0)
    perp = np.array([-mean_d[1], mean_d[0]])
    if perp[1] < 0:
        perp = -perp
    cpt = np.array([w / 2, h / 2])

    def cross(l):
        A = np.array([l[1], -perp]).T
        t = np.linalg.solve(A, cpt - l[0])
        return t[1]
    lines.sort(key=lambda l: -cross(l))
    R = len(lines)
    W_m = lanes * lane_width_m
    pool_c = np.float64([[0, 0], [0, W_m], [length_m, W_m], [length_m, 0]])
    xs = np.linspace(0.3, length_m - 0.3, 50)
    xs_out = np.concatenate([np.linspace(-2.0, -0.7, 6), np.linspace(length_m + 0.7, length_m + 2.0, 6)])

    def layout(outer):
        first = 0.0 if outer else lane_width_m
        nn = lanes + 1 if outer else lanes - 1
        rope_y = [first + j * lane_width_m for j in range(nn)]
        empty_y = [(j + 0.5) * lane_width_m for j in range(lanes)] + ([] if outer else [0.0, W_m])
        return (np.array([[x, y] for y in rope_y for x in xs]),
                np.array([[x, y] for y in empty_y for x in xs]),
                np.array([[x, y] for y in rope_y for x in xs_out]))
    layouts = {o: layout(o) for o in (True, False)}

    water_small = cv2.resize(water, (w // 4, h // 4), interpolation=cv2.INTER_NEAREST) > 0

    def water_f1(Hm):
        poly = cv2.perspectiveTransform(_dense_outline().reshape(-1, 1, 2), Hm).reshape(-1, 2)
        poly = lens_distort(poly, k, size) / 4
        if not np.all(np.isfinite(poly)):
            return 0.0
        m_ = np.zeros(water_small.shape, np.uint8)
        cv2.fillPoly(m_, [np.int32(np.clip(poly, -4 * w, 4 * w))], 1)
        m_ = m_ > 0
        tp = (m_ & water_small).sum()
        prec, rec = tp / max(m_.sum(), 1), tp / max(water_small.sum(), 1)
        return float(2 * prec * rec / max(prec + rec, 1e-6))

    def _dense_outline(n=12):
        pts = []
        for (x0, y0), (x1, y1) in zip(pool_c, np.roll(pool_c, -1, 0)):
            for t in np.linspace(0, 1, n, endpoint=False):
                pts.append([x0 + (x1 - x0) * t, y0 + (y1 - y0) * t])
        return np.float64(pts)

    def score_H(Hm, outer):
        on, off, past = layouts[outer]
        f = lambda P: cv2.perspectiveTransform(P.reshape(-1, 1, 2).astype(np.float64), Hm).reshape(-1, 2)
        s_on, ok_on = support(f(on))
        s_off, ok_off = support(f(off))
        s_past, ok_past = support(f(past))
        # total supported rope length (per lane): predicting a real rope adds, a missing one
        # adds nothing; off-screen counts as missing
        inside = float((s_on * ok_on).sum()) / (len(xs) * lanes)
        empty = float(s_off[ok_off].mean()) if ok_off.any() else 0.0
        beyond = float(s_past[ok_past].mean()) if ok_past.any() else 0.0
        wf = water_f1(Hm)
        score_H.parts = (round(inside, 3), round(empty, 3), round(beyond, 3), round(wf, 3))
        return inside - 0.8 * empty - 0.5 * beyond + 1.0 * wf, inside

    # 4) hypotheses: which pool rope slot is each found rope (with / without outer ropes,
    #    allowing one or two stray lines to be dropped), for the best wall candidates.
    #    Homography from every rope/wall intersection; score against ALL rope evidence.
    hyps = []
    for outer in {"yes": [True], "no": [False]}.get(outer_ropes, [True, False]):
        N = lanes + 1 if outer else lanes - 1
        first_y = 0.0 if outer else lane_width_m
        for keep_n in range(min(R, N), max(1, min(R, N) - 2), -1):
            for keep in itertools.combinations(range(R), keep_n):
                if R - keep_n > 2:
                    continue
                for idx in itertools.combinations(range(N), keep_n):
                    for (wa_m, wa_n), (wb_m, wb_n) in itertools.product(walls_a[:2], walls_b[:2]):
                        src, dst = [], []
                        for li, i in zip(keep, idx):
                            p_, d_, _, _ = lines[li]
                            y = first_y + i * lane_width_m
                            src += [inter(p_, d_, wa_m, wa_n), inter(p_, d_, wb_m, wb_n)]
                            dst += [[0.0, y], [length_m, y]]
                        Hm, _ = cv2.findHomography(np.float64(dst), np.float64(src), 0)
                        if Hm is None:
                            continue
                        rep = cv2.perspectiveTransform(np.float64(dst).reshape(-1, 1, 2), Hm).reshape(-1, 2)
                        fit = float(np.median(np.linalg.norm(rep - np.float64(src), axis=1)))
                        if fit > 0.03 * w:
                            continue
                        sv, inside = score_H(Hm, outer)
                        hyps.append((sv - 0.05 * (R - keep_n), inside, outer, idx, Hm, score_H.parts))
    if not hyps:
        raise GeometryError("could not fit the pool layout - check lanes / length settings")
    hyps.sort(key=lambda t: -t[0])
    dbg_h = [(round(hh[0], 3), hh[5], hh[2], hh[3]) for hh in hyps[:6]]

    # 5) polish the best few against all the rope evidence
    from scipy.optimize import minimize
    best = None
    for sv, inside, outer, idx, Hm, _ in hyps[:4]:
        c0 = cv2.perspectiveTransform(pool_c.reshape(-1, 1, 2), Hm).reshape(-1, 2)

        def cost(x, outer=outer):
            Hx = cv2.getPerspectiveTransform(pool_c.astype(np.float32), x.reshape(4, 2).astype(np.float32))
            return -score_H(Hx, outer)[0]
        res = minimize(cost, c0.ravel(), method="Powell", options={"xtol": 0.3, "ftol": 1e-4, "maxiter": 2000})
        Hx = cv2.getPerspectiveTransform(pool_c.astype(np.float32), res.x.reshape(4, 2).astype(np.float32))
        sx, ix = score_H(Hx, outer)
        if sx < sv:
            Hx, sx, ix = Hm, sv, inside
        if best is None or sx > best[0]:
            best = (sx, ix, outer, Hx)
    score_v, inside, outer, Hm = best

    # orientation: default lane 1 = bottom of image (nearest the camera), start = left end
    oc = pool_c.copy()
    if flip_ends:
        oc[:, 0] = length_m - oc[:, 0]
    if flip_lanes:
        oc[:, 1] = W_m - oc[:, 1]
    cc = cv2.perspectiveTransform(oc.reshape(-1, 1, 2), Hm).reshape(-1, 2)
    cd = lens_distort(from_c(cc), k, size)
    corners = [[round(float(x) / w, 5), round(float(y) / h, 5)] for x, y in cd]
    N = lanes + 1 if outer else lanes - 1
    report = (f"found {R} of {N} lane ropes ({'with' if outer else 'without'} outer ropes); "
              f"lens k={k:.3f} (rope bend {before:.1f}px -> {after:.1f}px); "
              f"rope evidence {inside:.0%}")
    ropes_px = []
    for p, d, n, _ in lines:
        ts = np.linspace(-2 * w, 2 * w, 400)
        P = lens_distort(p + ts[:, None] * d, k, size)
        P = P[(P[:, 0] > 0) & (P[:, 0] < w) & (P[:, 1] > 0) & (P[:, 1] < h)]
        ropes_px.append(P.tolist())
    return {"corners": corners, "lens": {"k": round(k, 4)}, "ropes": R, "report": report,
            "outer_ropes": bool(outer), "quality": round(float(inside), 3),
            "debug": {"ropes_px": ropes_px, "hyps": dbg_h,
                      "ends_a": lens_distort(np.array(ends_a), k, size).tolist(),
                      "ends_b": lens_distort(np.array(ends_b), k, size).tolist(),
                      "wall_a": [lens_distort(np.array([ma + t * np.array([-na[1], na[0]])]), k, size)[0].tolist() for t in np.linspace(-h, h, 40)],
                      "wall_b": [lens_distort(np.array([mb + t * np.array([-nb[1], nb[0]])]), k, size)[0].tolist() for t in np.linspace(-h, h, 40)]}}
