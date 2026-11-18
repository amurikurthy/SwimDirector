"""
Learned PTZ geometry: where must the PTZ point (pan/tilt) to look at a pool point?

Model: the PTZ is a rigid camera somewhere around the pool. Its pan/tilt readings are
spherical angles (fixed units/degree, unknown signs and tilt zero), mounted with an
unknown rotation at an unknown position. Given observations

    pool point (x, y)  <->  camera pan/tilt that looked straight at it

the mount is recovered with a perspective-n-point solve on a virtual pinhole camera.

Observations arrive as "frames" of candidate pairs: when the PTZ sees splash blobs and
the overview knows where swimmers are, any blob could be any swimmer. RANSAC picks one
candidate per frame and keeps the mount that explains the most frames, so the system
calibrates itself from warm-ups or from watching the operator work.
"""
import json
import logging
import math
import os
import random
import threading
import time
from collections import deque

import cv2
import numpy as np

log = logging.getLogger("swim.ptzmodel")
F = 1000.0                 # virtual focal length (px) for the PnP solve
K = np.float64([[F, 0, 0], [0, F, 0], [0, 0, 1]])


def _rot_to_z(v):
    """Rotation matrix mapping unit vector v onto +z."""
    v = v / np.linalg.norm(v)
    z = np.array([0.0, 0.0, 1.0])
    c = float(v @ z)
    if c > 0.999999:
        return np.eye(3)
    if c < -0.999999:
        return np.diag([1.0, -1.0, -1.0])
    ax = np.cross(v, z)
    s = np.linalg.norm(ax)
    ax /= s
    Kx = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    return np.eye(3) + s * Kx + (1 - c) * (Kx @ Kx)


class PTZModel:
    def __init__(self, path, units_per_deg=121.4):
        self.path = path
        self.upd = units_per_deg
        self.frames = deque(maxlen=900)  # each: {"t", "w", "cands": [[x, y, pan, tilt], ...]}
        self.teach = []                  # strong single-candidate frames from the operator
        self.params = None               # fitted mount
        self.k = (1, 1)                  # learned picture-direction signs (k_pan, k_tilt)
        self.stats = {"state": "not calibrated"}
        self._dirty = 0
        self._busy = False
        self._lock = threading.Lock()
        self.load()

    # ------------------------------------------------------------ persistence
    def load(self):
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path) as f:
                d = json.load(f)
            self.teach = d.get("teach", [])
            for fr in d.get("frames", []):
                self.frames.append(fr)
            p = d.get("params")
            if p:
                self.params = {k: (np.array(v) if isinstance(v, list) else v) for k, v in p.items()}
            self.stats = d.get("stats", self.stats)
        except Exception as e:
            log.warning("could not load %s: %s", self.path, e)

    def save(self):
        with self._lock:
            d = {"teach": self.teach, "frames": list(self.frames)[-600:], "stats": self.stats,
                 "params": {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                            for k, v in (self.params or {}).items()} or None}
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(d, f)
        os.replace(tmp, self.path)

    def reset(self):
        with self._lock:
            self.frames.clear()
            self.teach = []
            self.params = None
            self.stats = {"state": "not calibrated"}
        self.save()

    # ------------------------------------------------------------ data in
    def add_frame(self, cands, weight=1.0, t=None):
        """cands: list of (x, y, pan, tilt[, blob_track_id, lane]) - one of them (or none)
        is a true match. Track ids let the fit chain the same blob/swimmer over time."""
        if not cands:
            return
        with self._lock:
            self.frames.append({"t": t or time.time(), "w": weight,
                                "cands": [[round(float(a), 3) for a in c] for c in cands[:40]]})
            self._dirty += 1

    def add_teach(self, x, y, pan, tilt):
        with self._lock:
            self.teach.append({"t": time.time(), "w": 6.0, "cands": [[x, y, pan, tilt]]})
            self._dirty += 25

    def undo_teach(self):
        with self._lock:
            if self.teach:
                self.teach.pop()
                self._dirty += 25

    # ------------------------------------------------------------ geometry
    def _dirs(self, pan, tilt, sp, st, t0, ox=None, oy=None):
        """Unit directions in the PTZ's base frame. ox/oy: the blob's offset from the
        picture centre on the tangent plane (right / down, tan units)."""
        az = np.radians(sp * (np.asarray(pan, float) - 32768) / self.upd)
        el = np.radians(st * (np.asarray(tilt, float) - 32768) / self.upd + t0)
        ca, sa, ce, se = np.cos(az), np.sin(az), np.cos(el), np.sin(el)
        f = np.stack([ce * ca, ce * sa, se], -1)
        if ox is None:
            return f
        kp, kt = self.k
        right = (sp * kp) * np.stack([-sa, ca, np.zeros_like(sa)], -1)
        up = (st * -kt) * np.stack([-se * ca, -se * sa, ce], -1)
        d = f + np.asarray(ox, float)[..., None] * right - np.asarray(oy, float)[..., None] * up
        return d / np.linalg.norm(d, axis=-1, keepdims=True)

    def _units(self, d, p):
        """Base-frame direction -> (pan, tilt) camera units under fitted params p."""
        el = np.degrees(np.arcsin(np.clip(d[..., 2], -1, 1)))
        az = np.degrees(np.arctan2(d[..., 1], d[..., 0]))
        return 32768 + p["sp"] * az * self.upd, 32768 + p["st"] * (el - p["t0"]) * self.upd

    def observed_units(self, c):
        """The pan/tilt that would have centred candidate c (uses the blob offset)."""
        p = self.params
        c = (list(c) + [-1, -1, 0.0, 0.0])[:8]
        d = self._dirs(c[2], c[3], p["sp"], p["st"], p["t0"], c[6], c[7])
        u = self._units(d, p)
        az0 = np.degrees(np.arctan2(*self._dirs(c[2], c[3], p["sp"], p["st"], p["t0"])[[1, 0]]))
        pan = float(u[0])
        # keep the same wrap as the reading
        while pan - c[2] > 180 * self.upd:
            pan -= 360 * self.upd
        while c[2] - pan > 180 * self.upd:
            pan += 360 * self.upd
        return pan, float(u[1])

    def trusted(self):
        return bool(self.params) and self.stats.get("trusted", False)

    def predict(self, x, y):
        """-> (pan, tilt, distance_m) or None."""
        p = self.params
        if not p:
            return None
        Pw = np.array([x, y, 0.0])
        c = p["R"] @ Pw + p["t"]
        dist = float(np.linalg.norm(c))
        d = p["Q"].T @ (c / dist)
        el = math.degrees(math.asin(max(-1.0, min(1.0, d[2]))))
        az = math.degrees(math.atan2(d[1], d[0]))
        pan = 32768 + p["sp"] * az * self.upd
        tilt = 32768 + p["st"] * (el - p["t0"]) * self.upd
        return pan, tilt, dist

    def ray_to_pool(self, pan, tilt):
        """Where on the water a pan/tilt direction lands -> (x, y) or None (above horizon)."""
        p = self.params
        if not p:
            return None
        d = self._dirs(pan, tilt, p["sp"], p["st"], p["t0"])
        v = p["Q"] @ d                      # virtual-camera coordinates
        C = -p["R"].T @ p["t"]
        r = p["R"].T @ v                    # world direction
        if r[2] >= -1e-6:
            return None
        lam = -C[2] / r[2]
        P = C + lam * r
        return float(P[0]), float(P[1])

    def camera_position(self):
        p = self.params
        if not p:
            return None
        C = -p["R"].T @ p["t"]
        return [round(float(v), 2) for v in C]

    def pick(self, cands, tol_deg=2.5):
        """With a trusted model: the candidate the model explains (or None)."""
        if not self.trusted():
            return None
        best = None
        for c in cands:
            pr = self.predict(c[0], c[1])
            if pr is None:
                continue
            op, ot = self.observed_units(c)
            e = math.hypot(pr[0] - op, pr[1] - ot) / self.upd
            if e < tol_deg and (best is None or e < best[0]):
                best = (e, c)
        return best[1] if best else None

    # ------------------------------------------------------------ fitting
    def maybe_refit(self, force=False):
        if self._busy:
            return
        if not force and self._dirty < 40:
            return
        self._busy = True
        threading.Thread(target=self._refit_safe, daemon=True, name="ptz-fit").start()

    def _refit_safe(self):
        try:
            self.refit()
        except Exception as e:
            log.exception("PTZ model fit failed: %s", e)
        finally:
            self._busy = False

    def refit(self, thr_deg=2.0):
        with self._lock:
            frames = [fr for fr in list(self.teach) + list(self.frames) if fr["cands"]]
            self._dirty = 0
        if len(frames) < 4:
            self.stats = {"state": f"collecting ({len(frames)} samples)", "trusted": False}
            return
        t_start = time.time()
        allc = np.array([(list(c) + [-1, -1, 0.0, 0.0])[:8] for fr in frames for c in fr["cands"]], float)
        fid = np.concatenate([[i] * len(fr["cands"]) for i, fr in enumerate(frames)])
        wts = np.array([fr.get("w", 1.0) for fr in frames])
        P3 = np.c_[allc[:, :2], np.zeros(len(allc))]
        lo, hi = allc[:, :2].min(0), allc[:, :2].max(0)
        box = (lo[0] - 25, lo[1] - 25, hi[0] + 25, hi[1] + 25)   # PTZ is within 25 m of the pool
        thr = F * math.tan(math.radians(thr_deg))
        n_single = sum(1 for fr in frames if len(fr["cands"]) == 1)
        avg_c = float(np.mean([len(fr["cands"]) for fr in frames]))
        # more swimmers in view = more ways to be fooled = more tries needed
        iters = 400 if n_single >= 6 else int(np.clip(30 * avg_c ** 2, 600, 4000))

        def build(sp, st, t0):
            D = self._dirs(allc[:, 2], allc[:, 3], sp, st, t0, allc[:, 6], allc[:, 7])
            Q = _rot_to_z(D.mean(0))
            V = D @ Q.T
            ok = V[:, 2] > 0.2
            img = np.where(ok[:, None], F * V[:, :2] / np.maximum(V[:, 2:3], 1e-6), 1e9)
            return Q, img, ok

        # blob tracks: a splash followed over time must be explained by ONE swimmer for
        # most of its life. Random mounts explain single frames by luck, not whole tracks.
        tr_raw = allc[:, 4].astype(np.int64)
        tr = np.where(tr_raw >= 0, tr_raw, -1 - fid)            # untracked: own track per frame
        lane = allc[:, 5].astype(np.int64)
        _, tr_idx = np.unique(tr, return_inverse=True)
        n_tr = tr_idx.max() + 1
        tl_key = tr_idx * 64 + (lane % 64)
        _, tl_idx = np.unique(tl_key, return_inverse=True)
        tl_track = np.zeros(tl_idx.max() + 1, np.int64)
        tl_track[tl_idx] = tr_idx
        frames_per_track = np.zeros(n_tr)
        first_c = np.zeros(len(frames), np.int64)
        first_c[fid[::-1]] = np.arange(len(fid))[::-1]          # one candidate per frame
        np.add.at(frames_per_track, tr_idx[first_c], 1.0)

        def score(R, t, img, ok):
            Xc = P3 @ R.T + t
            z = Xc[:, 2]
            good = (z > 0.1) & ok
            uv = F * Xc[:, :2] / np.where(good, z, 1.0)[:, None]
            err = np.where(good, np.linalg.norm(uv - img, axis=1), 1e9)
            best = np.full(len(frames), 1e9)
            np.minimum.at(best, fid, err)
            hit = err < thr
            # per (track, lane): in how many frames does that swimmer explain the blob?
            tl_hits = np.zeros(len(tl_track))
            np.add.at(tl_hits, tl_idx[hit], wts[fid[hit]])
            best_lane = np.zeros(n_tr)
            np.maximum.at(best_lane, tl_track, tl_hits)
            consistent = best_lane >= np.maximum(1.0, 0.6 * frames_per_track)
            inl = (best < thr) & consistent[tr_idx[first_c]]
            return float(best_lane[consistent].sum()), best, inl

        def ransac(sp, st, t0, n_iter):
            Q, img, ok = build(sp, st, t0)
            if ok.sum() < 4:
                return None
            groups = [np.flatnonzero((fid == i) & ok) for i in range(len(frames))]
            groups = [g for g in groups if len(g)]
            if len(groups) < 4:
                return None
            gw = np.array([wts[fid[g[0]]] for g in groups], float)
            gw = gw / gw.sum()
            # chains: the same PTZ blob track paired with the same lane over time; two
            # chains in different lanes give a well-spread 4-point sample that is right
            # far more often than 4 independent guesses
            chains = {}
            for j in np.flatnonzero(ok):
                if allc[j, 4] >= 0 and allc[j, 5] >= 0:
                    chains.setdefault((int(allc[j, 4]), int(allc[j, 5])), []).append(j)
            chains = [np.array(v) for v in chains.values()
                      if len(v) >= 3 and np.ptp(allc[v, 0]) >= 1.5]
            best = None
            for it in range(n_iter):
                idx = None
                if len(chains) >= 2 and it % 4 != 3:
                    a, b = random.sample(range(len(chains)), 2)
                    ca, cb = chains[a], chains[b]
                    if allc[ca[0], 5] == allc[cb[0], 5]:
                        continue
                    ia = ca[np.argsort(allc[ca, 0])]
                    ib = cb[np.argsort(allc[cb, 0])]
                    idx = [ia[0], ia[-1], ib[0], ib[-1]]
                if idx is None:
                    gs = np.random.choice(len(groups), 4, replace=False, p=gw)
                    idx = [random.choice(list(groups[g])) for g in gs]
                o, im = P3[idx].astype(np.float64), img[idx].astype(np.float64)
                try:
                    n, rvecs, tvecs = cv2.solveP3P(o[:3], im[:3], K, None, flags=cv2.SOLVEPNP_AP3P)
                except cv2.error:
                    continue
                for rv, tv in zip(rvecs or [], tvecs or []):
                    R, _ = cv2.Rodrigues(rv)
                    tt = tv.ravel()
                    C = -R.T @ tt               # camera must be above the water, near the pool
                    if not (1.0 < C[2] < 25 and box[0] <= C[0] <= box[2] and box[1] <= C[1] <= box[3]):
                        continue
                    x4 = R @ o[3] + tt
                    if x4[2] <= 0 or np.linalg.norm(F * x4[:2] / x4[2] - im[3]) > 3 * thr:
                        continue
                    s, _, inl = score(R, tt, img, ok)
                    if best is None or s > best[0]:
                        best = (s, R, tt, inl)
            if best is None:
                return None
            # refine on inliers (best candidate per inlier frame)
            s, R, tt, inl = best
            for _ in range(3):
                Xc = P3 @ R.T + tt
                uv = F * Xc[:, :2] / np.maximum(Xc[:, 2:3], 1e-6)
                err = np.linalg.norm(uv - img, axis=1)
                sel = []
                for i in np.flatnonzero(inl):
                    m = fid == i
                    j = np.flatnonzero(m)[np.argmin(err[m])]
                    sel.append(j)
                if len(sel) < 4:
                    break
                rv, _ = cv2.Rodrigues(R)
                okp, rv, tv = cv2.solvePnP(P3[sel], img[sel], K, None, rv, tt.reshape(3, 1),
                                           useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)
                if not okp:
                    break
                R2, _ = cv2.Rodrigues(rv)
                t2 = tv.ravel()
                C2 = -R2.T @ t2
                if not (1.0 < C2[2] < 25 and box[0] <= C2[0] <= box[2] and box[1] <= C2[1] <= box[3]):
                    break
                s2, b2, inl2 = score(R2, t2, img, ok)
                if s2 < s:
                    break
                R, tt, s, bests, inl = R2, t2, s2, b2, inl2
            _, bests, inl = score(R, tt, img, ok)
            med = float(np.degrees(np.arctan(np.median(bests[inl]) / F))) if inl.any() else 99.0
            return {"score": s, "R": R, "t": tt, "Q": Q, "sp": sp, "st": st, "t0": t0,
                    "inliers": int(inl.sum()), "median_deg": med, "inl": inl}

        # coarse search over signs and tilt zero, then refine the tilt zero
        # an upright camera turns the picture opposite to its pan/tilt; try that first
        kp, kt = self.k
        sign_sets = [(-kp, -kt)]
        results = []
        for attempt in range(2):
            for sp, st in sign_sets:
                for t0 in range(-90, 91, 15):
                    r = ransac(sp, st, float(t0), max(150, iters // 3))
                    if r:
                        results.append(r)
            best_now = max(results, key=lambda r: r["score"]) if results else None
            total = float(wts.sum())
            if best_now and best_now["score"] >= 0.3 * total:
                break
            sign_sets = [(a, b) for a in (1, -1) for b in (1, -1) if (a, b) != (-kp, -kt)]
        if not results:
            self.stats = {"state": "no consistent fit yet", "trusted": False, "samples": len(frames)}
            return
        results.sort(key=lambda r: (-r["score"], r["median_deg"]))
        top = results[0]
        cpos = lambda r: -r["R"].T @ r["t"]
        rivals = [r for r in results[1:] if np.linalg.norm(cpos(r) - cpos(top)) > 3.0]
        second = rivals[0]["score"] if rivals else 0.0
        for t0 in np.arange(top["t0"] - 12, top["t0"] + 12.1, 3):
            r = ransac(top["sp"], top["st"], float(t0), iters)
            if r and (r["score"], -r["median_deg"]) > (top["score"], -top["median_deg"]):
                top = r
        # inlier spread (a fit from one small patch can't be trusted far away)
        inl_frames = int(top["inliers"])
        pts = np.array([c[:2] for fr, ok in zip(frames, top["inl"]) if ok for c in fr["cands"]])
        spread = float(np.ptp(pts[:, 0]) if len(pts) else 0.0)
        lanes_seen = len({round(c[1], 1) for fr, ok in zip(frames, top["inl"]) if ok for c in fr["cands"]
                          if len(fr["cands"]) == 1}) if n_single else 0
        C = -top["R"].T @ top["t"]
        plausible = 1.0 < C[2] < 25 and box[0] <= C[0] <= box[2] and box[1] <= C[1] <= box[3]
        teach_only = n_single >= 4 and inl_frames >= min(n_single, 4) and len(self.teach) >= 4
        unique = top["score"] >= 1.5 * second
        trusted = plausible and top["median_deg"] < 1.6 and spread >= 4.0 and (
            teach_only or (inl_frames >= 50 and inl_frames >= 0.5 * len(frames) and unique))
        p = {k: top[k] for k in ("R", "t", "Q", "sp", "st", "t0")}
        with self._lock:
            self.params = p
            C = -top["R"].T @ top["t"]
            self.stats = {
                "state": "calibrated" if trusted else ("learning - need more data" if plausible else "learning"),
                "trusted": bool(trusted),
                "samples": len(frames), "inliers": inl_frames,
                "median_err_deg": round(top["median_deg"], 2),
                "spread_m": round(spread, 1),
                "unique": round(float(top["score"] / max(second, 1e-6)), 2),
                "camera_at": [round(float(v), 1) for v in C],
                "fit_s": round(time.time() - t_start, 1),
                "updated": time.time(),
            }
        log.info("PTZ model: %s", self.stats)
        self.save()
