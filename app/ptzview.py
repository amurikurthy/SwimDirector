"""
PTZ feed vision: where are swimmers in the PTZ's own shot?

- Camera motion removed per frame (phase correlation); splash = white + moving pixels next
  to water, accumulated over a few frames.
- Returns every splash blob (for learning) and the one nearest centre (for fine-centring).
- Field of view is learned per zoom band from image shift vs. the camera's reported pan.
"""
import math

import cv2
import numpy as np

from .camera import ZOOM_MAX, ZOOM_MIN

DEFAULTS = dict(
    work_width=320,
    water_hue=[75, 115], water_min_sat=35, water_min_val=40,
    white_min_val=170, white_max_sat=70, diff_threshold=22,
    activity_decay=0.6, activity_threshold=1.2,
    min_blob_area=10, max_offset_x=0.3, max_offset_y=0.3,
    fov_wide_deg=62.0, zoom_ratio=20.0,   # starting guess only; learned while panning
)


def zoom_band(z):
    return int(np.clip(((z if z is not None else ZOOM_MIN) - ZOOM_MIN) / (ZOOM_MAX - ZOOM_MIN) * 16, 0, 15))


class PTZView:
    def __init__(self, units_per_deg=121.4, fov=None, cfg=None):
        self.c = {**DEFAULTS, **(cfg or {})}
        self.upd = units_per_deg
        self.fov = {int(k): float(v) for k, v in (fov or {}).items()}  # band -> frame width (pan units)
        self.prev_f = self.prev_u8 = self.win = self.activity = None
        self.size = None
        self._k = {}
        self._acc_dx = 0.0
        self._acc_bad = False
        self._ref = None
        self._est = {}            # band -> recent estimates (median is used)
        self.k_pan = None         # learned sign (set by the director)
        self.blobs = []           # [(ex, ey, area)] normalized offsets -0.5..0.5
        self.last = None          # blob chosen for fine-centring
        self.shift_px = (0.0, 0.0)
        self.shift_ok = False

    def _kern(self, n):
        if n not in self._k:
            self._k[n] = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (n, n))
        return self._k[n]

    # -------------------------------------------------------------- field of view
    def fov_units(self, z):
        """Horizontal frame width in pan units at zoom z (learned, else a wide-lens guess)."""
        b = zoom_band(z)
        if b in self.fov:
            return self.fov[b]
        learned = sorted(self.fov.items())
        if len(learned) >= 2:  # log-linear interpolation between learned bands
            bs = np.array([k for k, _ in learned], float)
            vs = np.log([v for _, v in learned])
            return float(np.exp(np.interp(b, bs, vs, left=None, right=None)))
        deg = self.c["fov_wide_deg"] / (self.c["zoom_ratio"] ** (b / 15.0))
        return deg * self.upd

    def learn_fov(self, pos, zoom):
        """Compare how far the image slid with how far the camera says it panned."""
        if pos is None:
            return None
        b = zoom_band(zoom)
        if self._ref is None or self._ref[1] != b or abs(pos[2] - self._ref[0][2]) > 3 or self._acc_bad:
            self._ref, self._acc_dx, self._acc_bad = (pos, b), 0.0, False
            return None
        if pos[2] == self._ref[0][2]:
            return None
        dpan = pos[0] - self._ref[0][0]
        dtilt = pos[1] - self._ref[0][1]
        if abs(dpan) > 250 and abs(self._acc_dx) > 0.04 and abs(dtilt) < 0.3 * abs(dpan):
            est = abs(dpan / self._acc_dx)
            consistent = self.k_pan is None or math.copysign(1, dpan) == self.k_pan * math.copysign(1, -self._acc_dx)
            if not consistent or not (0.8 * self.upd < est < 110 * self.upd):
                self._ref, self._acc_dx = (pos, b), 0.0
                return None
            lst = self._est.setdefault(b, [])
            lst.append(est)
            del lst[:-15]
            self.fov[b] = float(np.median(lst))
            self._ref, self._acc_dx = (pos, b), 0.0
            return b
        return None

    def tangent(self, ex, ey, z):
        """Image offset -> tangent-plane offsets (right, down) for exact geometry."""
        hf = math.radians(self.fov_units(z) / self.upd)
        aspect = (self.size[1] / self.size[0]) if self.size else 0.5625
        t = 2 * math.tan(hf / 2)
        return round(ex * t, 5), round(ey * t * aspect, 5)

    def offset_to_units(self, ex, ey, z, k_pan=1, k_tilt=1):
        """Image offset (normalized) -> (dpan, dtilt) camera units needed to centre it.
        k_pan / k_tilt: learned signs (camera value change per image direction)."""
        fu = self.fov_units(z)
        aspect = (self.size[1] / self.size[0]) if self.size else 0.5625
        return k_pan * ex * fu, k_tilt * ey * fu * aspect

    # -------------------------------------------------------------- detection
    def process(self, frame):
        c = self.c
        h0, w0 = frame.shape[:2]
        W = int(c["work_width"])
        H = int(round(h0 * W / w0))
        if self.size != (W, H):
            self.size = (W, H)
            self.prev_f = self.prev_u8 = None
            self.activity = np.zeros((H, W), np.float32)
            self.win = cv2.createHanningWindow((W, H), cv2.CV_32F)
        small = cv2.resize(frame, (W, H), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        gf = gray.astype(np.float32)
        if float(gray.std()) < 4.0:      # featureless (sky, ceiling, blank): can't measure motion
            self._acc_bad = True
        cand = None
        self.shift_ok = False
        if self.prev_f is not None:
            (dx, dy), resp = cv2.phaseCorrelate(self.prev_f, gf, self.win)
            self.shift_ok = resp > 0.03 and abs(dx) < 0.3 * W and abs(dy) < 0.3 * H
            if not self.shift_ok:
                self._acc_bad = True
            if self.shift_ok:
                self.shift_px = (dx, dy)
                self._acc_dx += dx / W
                M = np.float32([[1, 0, dx], [0, 1, dy]])
                prev_al = cv2.warpAffine(self.prev_u8, M, (W, H), borderMode=cv2.BORDER_REPLICATE)
                diff = cv2.absdiff(cv2.GaussianBlur(gray, (3, 3), 0), cv2.GaussianBlur(prev_al, (3, 3), 0))
                water = cv2.inRange(hsv, (c["water_hue"][0], c["water_min_sat"], c["water_min_val"]),
                                    (c["water_hue"][1], 255, 255))
                near_water = cv2.dilate(water, self._kern(9)) > 0
                white = (hsv[..., 2] >= c["white_min_val"]) & (hsv[..., 1] <= c["white_max_sat"])
                m = (white & (diff >= c["diff_threshold"])) | (diff >= 2 * c["diff_threshold"])
                cand = (m & near_water).astype(np.float32)
                self.activity = cv2.warpAffine(self.activity, M, (W, H))
        self.prev_f, self.prev_u8 = gf, gray
        self.activity *= c["activity_decay"]
        if cand is not None:
            self.activity += cand
        self.blobs, self.last = [], None
        if not self.shift_ok:  # zooming / whip pan: not trustworthy
            return None
        bm = (self.activity >= c["activity_threshold"]).astype(np.uint8) * 255
        bm = cv2.morphologyEx(bm, cv2.MORPH_CLOSE, self._kern(5))
        n, lab, stats, cents = cv2.connectedComponentsWithStats(bm)
        min_area = c["min_blob_area"] * (W / 320) ** 2
        best, best_score = None, 0.0
        for i in range(1, n):
            a = float(stats[i, cv2.CC_STAT_AREA])
            if a < min_area or a > 0.15 * W * H:
                continue
            ex, ey = cents[i][0] / W - 0.5, cents[i][1] / H - 0.5
            self.blobs.append((float(ex), float(ey), a))
            if abs(ex) <= c["max_offset_x"] and abs(ey) <= c["max_offset_y"]:
                score = a / (1.0 + 25 * (ex * ex + ey * ey))
                if score > best_score:
                    best, best_score = (float(ex), float(ey)), score
        self.last = best
        return best

    # -------------------------------------------------------------- drawing
    def draw(self, frame, lock=False, note=None):
        h, w = frame.shape[:2]
        cv2.drawMarker(frame, (w // 2, h // 2), (255, 255, 255), cv2.MARKER_CROSS, 40, 1)
        for ex, ey, a in self.blobs:
            p = (int((ex + 0.5) * w), int((ey + 0.5) * h))
            cv2.circle(frame, p, 8, (255, 180, 0), 1)
        if self.last:
            p = (int((self.last[0] + 0.5) * w), int((self.last[1] + 0.5) * h))
            col = (0, 255, 0) if lock else (0, 200, 255)
            cv2.circle(frame, p, 20, col, 2)
            cv2.arrowedLine(frame, (w // 2, h // 2), p, col, 1, tipLength=0.15)
        if note:
            _label(frame, note, (10, 28))
        return frame


def _label(img, text, org, col=(255, 255, 255), scale=0.6):
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    x, y = org
    cv2.rectangle(img, (x - 4, y - th - 6), (x + tw + 4, y + base + 2), (0, 0, 0), -1)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, col, 1, cv2.LINE_AA)
