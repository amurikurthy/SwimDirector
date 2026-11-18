"""
Overview (static birdseye) vision.

The 4 clicked pool corners give a homography to a top-down "rectified" pool where
every lane is a horizontal band of equal height. Swimmers are found per lane as
splash/motion activity, tracked along the lane in metres, with lap counting so the
leader stays correct through turns.

Pool coordinates: x = metres from the start wall (corners 1-2) toward the far wall,
y = metres across from the lane-1 side (corners 1-4).
Corner click order: 1 start wall / lane-1 side, 2 start wall / last-lane side,
                    3 far wall / last-lane side, 4 far wall / lane-1 side.
"""
from dataclasses import dataclass

import cv2
import numpy as np

DEFAULTS = dict(
    px_per_m=12, lane_px=24,
    white_min_val=170, white_max_sat=70, diff_threshold=20,
    activity_decay=0.7, rope_margin=0.18,
    glare_rate=0.35, glare_window_s=20.0,  # pixels "splashing" >35% of the time = glare
    col_threshold=3.0, min_mass=25.0, merge_gap_m=1.5, min_len_m=0.3,
    gate_m=5.0, fresh_s=0.6, alive_s=8.0, race_reset_s=20.0, min_hits=12,
    min_speed=0.3, turn_zone_m=5.0, underwater_speed=0.8,
    leader_margin_m=0.7, leader_hold_s=0.6,
)


# ---------------------------------------------------------------------- lens (division model)
# Radius is normalized by half the image diagonal. undistorted = distorted / (1 + k r_d^2);
# k < 0 corrects barrel (wide-angle) distortion and stays invertible over the whole frame.
def _norm(size):
    w, h = size
    return np.array([w / 2, h / 2]), 0.5 * float(np.hypot(w, h))


def lens_undistort(pts, k, size):
    pts = np.float64(pts).reshape(-1, 2)
    if not k:
        return pts
    c, R = _norm(size)
    q = (pts - c) / R
    r2 = (q * q).sum(1, keepdims=True)
    return q / (1 + k * r2) * R + c


def lens_distort(pts, k, size):
    pts = np.float64(pts).reshape(-1, 2)
    if not k:
        return pts
    c, R = _norm(size)
    q = (pts - c) / R
    ru = np.sqrt((q * q).sum(1, keepdims=True))
    ru_safe = np.where(ru < 1e-9, 1e-9, ru)
    disc = np.maximum(1 - 4 * k * ru_safe ** 2, 0)
    rd = (1 - np.sqrt(disc)) / (2 * k * ru_safe)
    return q * (rd / ru_safe) * R + c


def fit_lens(lines_px, size):
    """lines_px: list of point lists, each along one straight lane rope.
    Returns (k, residual_px_before, residual_px_after)."""
    def resid(k):
        tot = []
        for pts in lines_px:
            u = lens_undistort(pts, k, size)
            ctr = u - u.mean(0)
            s = np.linalg.svd(ctr, compute_uv=False)
            tot.append(s[-1] / np.sqrt(len(u)))
        return float(np.mean(tot))
    ks = np.linspace(-0.6, 0.3, 181)
    best = min(ks, key=resid)
    fine = np.linspace(best - 0.005, best + 0.005, 21)
    best = float(min(fine, key=resid))
    return best, resid(0.0), resid(best)


@dataclass
class LaneState:
    idx: int          # 0-based from the lane-1 side
    x: float          # metres from start wall (smoothed / predicted)
    mx: float         # last measured x
    mt: float         # time of last measurement
    lo: float = 0.0
    hi: float = 0.0
    v: float = 0.0    # m/s, + = toward far wall
    dir: int = 0
    laps: int = 0
    hits: int = 0
    wall_t: float = -1e9
    pred_turn_t: float = -1e9
    run: float = 0.0          # metres travelled since the last turn / start


class Overview:
    def __init__(self, corners, length_m, lanes, lane_width_m=2.5, lane_first=1, cfg=None,
                 ignore=None, lens=None):
        self.c = {**DEFAULTS, **(cfg or {})}
        self.lens = lens or {}  # {"k": division-model distortion (<0 = barrel)}
        self.ignore = ignore or []  # [x1, y1, x2, y2] normalized image rects to ignore
        self.corners = np.float32(corners)  # normalized 0..1, 4x2
        self.L = float(length_m)
        self.lanes = int(lanes)
        self.lw = float(lane_width_m)
        self.lane_first = int(lane_first)
        self.RW = int(self.L * self.c["px_per_m"])
        self.lp = int(self.c["lane_px"])
        self.RH = self.lanes * self.lp
        self.size = None
        self.prev = None
        self.activity = None
        self.state: dict[int, LaneState] = {}
        self.leader_idx = None
        self._chal, self._chal_t = None, 0.0
        self.t_prev = None
        m = int(self.c["rope_margin"] * self.lp)
        mask = np.zeros((self.RH, self.RW), np.float32)
        for i in range(self.lanes):
            mask[i * self.lp + m:(i + 1) * self.lp - m] = 1.0
        self.lane_mask = mask
        self.mask = mask

    # ------------------------------------------------------------------ geometry
    def _setup(self, w, h):
        self.size = (w, h)
        self.lens_k = float(self.lens.get("k", 0.0))
        src = np.float32(self.undistort(self.corners * np.float32([w, h])))
        dst = np.float32([[0, 0], [0, self.RH], [self.RW, self.RH], [self.RW, 0]])
        self.M = cv2.getPerspectiveTransform(src, dst)
        self.Minv = cv2.getPerspectiveTransform(dst, src)
        # one remap: rectified pool pixel -> raw (distorted) image pixel
        gx, gy = np.meshgrid(np.arange(self.RW, dtype=np.float32) + 0.5,
                             np.arange(self.RH, dtype=np.float32) + 0.5)
        und = cv2.perspectiveTransform(np.stack([gx, gy], -1).reshape(-1, 1, 2), self.Minv)
        raw = self.distort(und.reshape(-1, 2)).reshape(self.RH, self.RW, 2).astype(np.float32)
        self.mapx, self.mapy = raw[..., 0].copy(), raw[..., 1].copy()
        self.prev = None
        self.activity = np.zeros((self.RH, self.RW), np.float32)
        self.hit_rate = np.zeros((self.RH, self.RW), np.float32)
        self.frames = 0
        self.mask = self.lane_mask
        if self.ignore:
            keep = np.ones((h, w), np.uint8)
            for x1, y1, x2, y2 in self.ignore:
                keep[int(min(y1, y2) * h):int(max(y1, y2) * h),
                     int(min(x1, x2) * w):int(max(x1, x2) * w)] = 0
            keep_r = cv2.remap(keep, self.mapx, self.mapy, cv2.INTER_NEAREST)
            self.mask = self.lane_mask * keep_r.astype(np.float32)

    def ensure(self, frame):
        h, w = frame.shape[:2]
        if self.size != (w, h):
            self._setup(w, h)

    def undistort(self, pts):
        return lens_undistort(pts, self.lens_k, self.size)

    def distort(self, pts):
        return lens_distort(pts, self.lens_k, self.size)

    def pool_to_img(self, x_m, y_m):
        p = np.float32([[[x_m * self.RW / self.L, y_m * self.RH / (self.lanes * self.lw)]]])
        q = self.distort(cv2.perspectiveTransform(p, self.Minv)[0])[0]
        return float(q[0]), float(q[1])

    def img_to_pool(self, px, py):
        u = np.float32(self.undistort([[px, py]])).reshape(1, 1, 2)
        q = cv2.perspectiveTransform(u, self.M)[0, 0]
        return float(q[0] * self.L / self.RW), float(q[1] * self.lanes * self.lw / self.RH)

    def _polyline(self, pts_m, n=30):
        """Pool-space segment(s) -> curved image polyline (follows lens distortion)."""
        out = []
        for (x0, y0), (x1, y1) in zip(pts_m, pts_m[1:]):
            for t in np.linspace(0, 1, n, endpoint=False):
                out.append(self.pool_to_img(x0 + (x1 - x0) * t, y0 + (y1 - y0) * t))
        out.append(self.pool_to_img(*pts_m[-1]))
        return np.int32(out)

    def lane_number(self, idx):
        return self.lane_first + idx

    def lane_center_m(self, idx):
        return (idx + 0.5) * self.lw

    # ------------------------------------------------------------------ main
    def reset_race(self):
        self.state = {}
        self.leader_idx = None
        self._chal = None

    def process(self, frame, now):
        c = self.c
        self.ensure(frame)
        dt = min(max(now - self.t_prev, 1e-3), 0.5) if self.t_prev else 1 / 30
        self.t_prev = now
        rect = cv2.remap(frame, self.mapx, self.mapy, cv2.INTER_LINEAR)
        self.rect = rect
        hsv = cv2.cvtColor(rect, cv2.COLOR_BGR2HSV)
        gray = cv2.GaussianBlur(cv2.cvtColor(rect, cv2.COLOR_BGR2GRAY), (3, 3), 0)
        cand = None
        if self.prev is not None:
            diff = cv2.absdiff(gray, self.prev)
            white = (hsv[..., 2] >= c["white_min_val"]) & (hsv[..., 1] <= c["white_max_sat"])
            m = (white & (diff >= c["diff_threshold"])) | (diff >= 2 * c["diff_threshold"])
            cand = m.astype(np.float32) * self.mask
            # long-run hit rate per pixel; glare/reflections are "on" far more often than
            # a swimmer passing through, so suppress pixels above glare_rate
            self.frames += 1
            a = max(dt / c["glare_window_s"], 1.0 / self.frames)
            self.hit_rate += a * (cand - self.hit_rate)
            cand *= self.hit_rate < c["glare_rate"]
        self.prev = gray
        self.activity *= c["activity_decay"]
        if cand is not None:
            self.activity += cand

        meas = self._measure()
        self._update_lanes(meas, now, dt)
        self._update_leader(now)

    def _measure(self):
        c = self.c
        ppm = c["px_per_m"]
        m = int(c["rope_margin"] * self.lp)
        out = {}
        for i in range(self.lanes):
            prof = self.activity[i * self.lp + m:(i + 1) * self.lp - m].sum(axis=0)
            prof = cv2.blur(prof.reshape(1, -1), (9, 1)).ravel()
            on = np.concatenate([[False], prof >= c["col_threshold"], [False]])
            edges = np.flatnonzero(np.diff(on.astype(np.int8)))
            runs = list(zip(edges[::2], edges[1::2]))  # [start, end)
            if not runs:
                continue
            merged = [list(runs[0])]
            for s, e in runs[1:]:
                if s - merged[-1][1] < c["merge_gap_m"] * ppm:
                    merged[-1][1] = e
                else:
                    merged.append([s, e])
            best, best_mass = None, 0.0
            for s, e in merged:
                mass = float(prof[s:e].sum())
                if mass > best_mass and (e - s) >= c["min_len_m"] * ppm:
                    best, best_mass = (s, e), mass
            if best is None or best_mass < c["min_mass"]:
                continue
            s, e = best
            xs = np.arange(s, e)
            xc = float((xs * prof[s:e]).sum() / prof[s:e].sum())
            out[i] = (xc / ppm, s / ppm, e / ppm)
        return out

    def _update_lanes(self, meas, now, dt):
        c = self.c
        L = self.L
        for ls in self.state.values():
            if now - ls.mt > c["fresh_s"]:  # coasting (e.g. underwater after a turn)
                if ls.pred_turn_t > ls.mt and now - ls.pred_turn_t > 3.0:
                    # predicted a turn but nobody resurfaced: it was a finish, undo
                    ls.laps = max(0, ls.laps - 1)
                    ls.dir = -ls.dir
                    ls.v = 0.0
                    ls.x = L if ls.dir > 0 else 0.0
                    ls.pred_turn_t = -1e9
                    continue
                ls.x += ls.v * dt
                if ls.dir and ls.v * ls.dir > c["min_speed"] and ls.run > 0.5 * L and \
                        ((ls.dir > 0 and ls.x >= L - 0.5) or (ls.dir < 0 and ls.x <= 0.5)):
                    ls.laps += 1
                    ls.dir = -ls.dir
                    ls.v = ls.dir * c["underwater_speed"]
                    ls.pred_turn_t = now
                    ls.run = 0.0
                ls.x = min(max(ls.x, 0.0), L)
        for i, (xc, lo, hi) in meas.items():
            ls = self.state.get(i)
            if ls is None:
                self.state[i] = LaneState(i, xc, xc, now, lo, hi, hits=1)
                continue
            gap = now - ls.mt
            if abs(xc - ls.x) > c["gate_m"] and gap < 0.5:
                continue
            if gap < 0.5:
                ls.v = 0.75 * ls.v + 0.25 * (xc - ls.mx) / max(gap, 1e-3)
                ls.run += abs(xc - ls.mx)
                ls.x = 0.6 * xc + 0.4 * ls.x
            else:
                ls.v = 0.0
                ls.x = xc
            ls.mx, ls.mt, ls.lo, ls.hi = xc, now, lo, hi
            ls.hits += 1
            if ls.x < c["turn_zone_m"] or ls.x > L - c["turn_zone_m"]:
                ls.wall_t = now
            if abs(ls.v) > c["min_speed"]:
                nd = 1 if ls.v > 0 else -1
                if ls.dir == 0:
                    ls.dir = nd
                elif nd != ls.dir:
                    if now - ls.pred_turn_t < 10:  # predicted a turn that didn't happen
                        ls.laps = max(0, ls.laps - 1)
                        ls.pred_turn_t = -1e9
                    elif now - ls.wall_t < 8 and ls.run > 0.5 * L:   # a real length, then a turn
                        ls.laps += 1
                    ls.dir = nd
                    ls.run = 0.0
        if self.state and all(now - s.mt > c["race_reset_s"] for s in self.state.values()):
            self.reset_race()

    def progress(self, ls):
        if ls.dir == 0:
            return None
        if ls.pred_turn_t > ls.mt:  # turn predicted but not yet confirmed: count up to the wall
            return ls.laps * self.L
        return ls.laps * self.L + (ls.x if ls.dir > 0 else self.L - ls.x)

    def _alive(self, ls, now):
        return now - ls.mt < self.c["alive_s"] and ls.hits >= self.c["min_hits"]

    def _update_leader(self, now):
        c = self.c
        cands = [s for s in self.state.values() if self._alive(s, now) and s.dir != 0]
        cur = self.state.get(self.leader_idx)
        if not cands:
            if cur is None or not self._alive(cur, now):
                self.leader_idx = None
            return
        best = max(cands, key=self.progress)
        if cur is None or not self._alive(cur, now) or cur.dir == 0:
            self.leader_idx, self._chal = best.idx, None
            return
        if best.idx != cur.idx and self.progress(best) - self.progress(cur) > c["leader_margin_m"]:
            if self._chal != best.idx:
                self._chal, self._chal_t = best.idx, now
            elif now - self._chal_t >= c["leader_hold_s"]:
                self.leader_idx, self._chal = best.idx, None
        else:
            self._chal = None

    # ------------------------------------------------------------------ targets
    def leader(self, now):
        s = self.state.get(self.leader_idx)
        return s if s and self._alive(s, now) else None

    def lane(self, number, now):
        s = self.state.get(number - self.lane_first)
        return s if s and self._alive(s, now) and now - s.mt < self.c["alive_s"] else None

    # ------------------------------------------------------------------ drawing
    def draw(self, frame, now, aim=None, debug=False):
        self.ensure(frame)
        h, w = frame.shape[:2]
        for x1, y1, x2, y2 in self.ignore:
            p1, p2 = (int(x1 * w), int(y1 * h)), (int(x2 * w), int(y2 * h))
            cv2.rectangle(frame, p1, p2, (128, 128, 128), 2)
            cv2.putText(frame, "ignore", (min(p1[0], p2[0]) + 4, min(p1[1], p2[1]) + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1)
        W_m = self.lanes * self.lw
        cv2.polylines(frame, [self._polyline([(0, 0), (0, W_m), (self.L, W_m), (self.L, 0),
                                              (0, 0)])], False, (255, 200, 0), 2)
        for k in range(1, self.lanes):
            cv2.polylines(frame, [self._polyline([(0, k * self.lw), (self.L, k * self.lw)])],
                          False, (255, 200, 0), 1)
        for i in range(self.lanes):
            p = self.pool_to_img(-0.8, self.lane_center_m(i))
            cv2.putText(frame, str(self.lane_number(i)), (int(p[0]) - 6, int(p[1]) + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)
        for s in self.state.values():
            if not self._alive(s, now):
                continue
            ld = s.idx == self.leader_idx
            col = (0, 255, 0) if ld else ((255, 180, 0) if now - s.mt < 0.6 else (150, 150, 150))
            p = tuple(map(int, self.pool_to_img(s.x, self.lane_center_m(s.idx))))
            cv2.circle(frame, p, 10 if ld else 7, col, 2)
            if s.dir:
                q = self.pool_to_img(s.x + 2.0 * s.dir, self.lane_center_m(s.idx))
                cv2.arrowedLine(frame, p, tuple(map(int, q)), col, 1, tipLength=0.4)
            cv2.putText(frame, f"L{self.lane_number(s.idx)} lap{s.laps}" + (" LEAD" if ld else ""),
                        (p[0] + 10, p[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)
        if aim is not None:
            p = tuple(map(int, self.pool_to_img(*aim)))
            cv2.drawMarker(frame, p, (0, 0, 255), cv2.MARKER_CROSS, 22, 2)
        if debug and self.activity is not None:
            a = np.clip(self.activity * 80, 0, 255).astype(np.uint8)
            a = cv2.applyColorMap(a, cv2.COLORMAP_INFERNO)
            h, w = frame.shape[:2]
            tw = min(w // 2, a.shape[1] * 2)
            th = int(a.shape[0] * tw / a.shape[1])
            frame[0:th, 0:tw] = cv2.resize(a, (tw, th))
        return frame
