"""
SwimDirector engine: watches both NDI feeds, understands the race, drives the PTZ,
tells Companion what's happening, and learns from what it sees and from the operator.
"""
import json
import logging
import math
import os
import threading
import time
from collections import deque

import cv2
import numpy as np

from . import autogeo
from .camera import ZOOM_MAX, ZOOM_MIN, MotionOut, PanasonicPTZ
from .companion import Companion
from .config import DATA_DIR, RESTART_KEYS
from .control import AxisController, clamp
from .overview import Overview
from .ptzmodel import PTZModel
from .ptzview import PTZView, _label
from .sources import NDIHub, VideoSource

log = logging.getLogger("swim.director")

PHASES = ("idle", "warmup", "ready", "race", "finish")


class Engine:
    def __init__(self, cfg):
        self.cfg = cfg
        self.lock = threading.RLock()
        self.think = deque(maxlen=200)          # (wall time, text, level)
        self.calib_path = os.path.join(DATA_DIR, "calib.json")
        self.calib = self._load_json(self.calib_path, {})
        self.learned = self.calib.setdefault("learned", {})
        self.hub = NDIHub(cfg.get("ndi", "discovery_server"))
        self.model = PTZModel(os.path.join(DATA_DIR, "ptzmodel.json"),
                              cfg.get("camera", "units_per_deg", default=121.4))
        self.src = {}
        self.ptz = None
        self.comp = None
        self.motion = None
        self.ov = None
        self.pview = None
        self.geo_job = None                     # {"frames": [...], "until": t}
        self.geo_bg = None
        self.autocal = None                     # sweep state
        self.stream_clients = {"overview": 0.0, "ptz": 0.0}
        self.jpeg = {"overview": None, "ptz": None}
        # director state
        self.force_track = None                 # None = auto, True/False = operator forced
        self.mode, self.lane = "leader", None
        self.zoom_bias = 0.0
        self.phase, self.phase_t = "idle", time.time()
        self.race = None
        self.quiet_since = None
        self.override_until = 0.0
        self.target_desc = None
        self.aim = None
        self.ptz_lock = False
        self.last_leader = None
        self._events = []
        self._btracks = []                      # PTZ blob tracks in camera angles
        self._btid = 0
        self._sample_t = 0.0
        self._cmd_hist = deque(maxlen=40)       # (t, raw_pan, raw_tilt)
        self._pos_hist = deque(maxlen=40)       # (t, pan, tilt)
        self._sign_ev = {"k_pan": 0.0, "k_tilt": 0.0, "c_pan": 0.0, "c_tilt": 0.0}
        self._ovr_hits = 0
        self._last_zoom_cmd = (None, 0.0)
        self._op_hold = None
        self._stop = False
        self.pan_ctl = AxisController(**cfg.get("control", "pan"))
        self.tilt_ctl = AxisController(**cfg.get("control", "tilt"))
        self.configure(initial=True)
        self.say("SwimDirector started.")
        threading.Thread(target=self._loop, daemon=True, name="engine").start()

    # ================================================================ utilities
    @staticmethod
    def _load_json(path, default):
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            return default

    def save_calib(self):
        tmp = self.calib_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.calib, f, indent=1)
        os.replace(tmp, self.calib_path)

    def say(self, text, level="info"):
        self.think.append((time.time(), text, level))
        log.info("%s", text)

    def event(self, name, why):
        """Director event -> Companion button (if mapped) + log."""
        loc = (self.cfg.get("companion", "buttons", default={}) or {}).get(name, "")
        if loc and self.comp:
            self.comp.press(loc, why)
        self._events.append((time.time(), name, why))
        self._events = self._events[-50:]

    # ================================================================ setup
    def configure(self, initial=False, changed=None):
        """(Re)build components from settings. NDI discovery changes need a restart."""
        c = self.cfg.snapshot()
        with self.lock:
            for key, sec in (("overview", "overview"), ("ptz", "ptz_feed")):
                want = {k: c[sec].get(k) for k in ("type", "name", "url")}
                cur = self.src.get(key)
                if cur is None or cur.cfg != want:
                    if cur:
                        cur.stop()
                    self.src[key] = VideoSource(key, want, self.hub)
            cam = c["camera"]
            if (self.ptz is None or self.ptz.host != cam["host"] or
                    (changed and any(k[0] == "camera" for k in changed))):
                if self.ptz:
                    self.ptz.close()
                self.ptz = PanasonicPTZ(cam["host"], cam.get("user"), cam.get("password"))
            comp = c["companion"]
            base = f"http://{comp['host']}:{int(comp.get('port') or 8000)}" if comp["host"] else None
            if self.comp is None or self.comp.base != base or self.comp.prefix != comp["variable_prefix"]:
                self.comp = Companion(comp["host"], comp.get("port", 8000), comp["variable_prefix"])
            self.motion = MotionOut(self.ptz, self.comp, cam.get("control", "direct"))
            self.pan_ctl = AxisController(**c["control"]["pan"])
            self.tilt_ctl = AxisController(**c["control"]["tilt"])
            self.model.upd = float(cam.get("units_per_deg", 121.4))
            if self.pview is None:
                self.pview = PTZView(self.model.upd, self.learned.get("fov"))
            self._build_overview()
        if changed and any(k in RESTART_KEYS for k in changed):
            return "restart"
        return None

    def _build_overview(self):
        g = self.calib.get("geometry")
        p = self.cfg.snapshot()["pool"]
        if not g or not g.get("corners"):
            self.ov = None
            return
        vis = dict(self.cfg.snapshot()["vision"])
        self.ov = Overview(g["corners"], p["length_m"], p["lanes"], p["lane_width_m"],
                           p["lane_first"], vis, g.get("ignore", []), g.get("lens"))

    # ================================================================ geometry
    def start_auto_geometry(self):
        self.geo_job = {"frames": [], "next": 0.0}
        self.say("Looking for the pool: collecting a few seconds of overview video…")

    def _geo_step(self, frame, now):
        job = self.geo_job
        if now < job["next"]:
            return
        job["frames"].append(cv2.resize(frame, (min(1280, frame.shape[1]),
                                                int(frame.shape[0] * min(1280, frame.shape[1]) / frame.shape[1]))))
        job["next"] = now + 0.3
        if len(job["frames"]) >= 15:
            frames = job["frames"]
            self.geo_job = None
            threading.Thread(target=self._geo_run, args=(frames,), daemon=True).start()

    def _geo_run(self, frames, bg=None):
        try:
            bg = bg if bg is not None else autogeo.background(frames)
            self.geo_bg = bg
            p = self.cfg.snapshot()["pool"]
            g0 = self.calib.get("geometry", {})
            r = autogeo.detect(bg, p["lanes"], p["lane_width_m"], p["length_m"], p.get("outer_ropes", "auto"),
                               g0.get("flip_lanes", False), g0.get("flip_ends", False))
            g = {"corners": r["corners"], "lens": r["lens"], "ignore": g0.get("ignore", []),
                 "flip_lanes": g0.get("flip_lanes", False), "flip_ends": g0.get("flip_ends", False),
                 "report": r["report"], "quality": r["quality"], "source": "auto", "time": time.time()}
            with self.lock:
                self.calib["geometry"] = g
                self.save_calib()
                self._build_overview()
            q = "looks solid" if r["quality"] > 0.45 else "is weak - check the outline and flip/adjust if needed"
            self.say(f"Pool found: {r['report']}. The fit {q}.")
            if self.cfg.get("pool", "outer_ropes") == "auto":
                pass
        except autogeo.GeometryError as e:
            self.say(f"Couldn't find the pool automatically: {e}. Set the 4 corners by hand on the "
                     "overview preview.", "warn")
        except Exception as e:
            log.exception("geometry")
            self.say(f"Pool detection failed: {e}", "warn")

    def flip_geometry(self, what):
        g = self.calib.setdefault("geometry", {})
        key = {"lanes": "flip_lanes", "ends": "flip_ends"}[what]
        g[key] = not g.get(key, False)
        c = g.get("corners")
        if c and len(c) == 4:
            # corner order: 1 start/lane1, 2 start/last, 3 far/last, 4 far/lane1
            c = [c[1], c[0], c[3], c[2]] if what == "lanes" else [c[3], c[2], c[1], c[0]]
            g["corners"] = c
        self.save_calib()
        self._build_overview()
        self.say(f"Flipped {'lane numbering' if what == 'lanes' else 'start/far end'}.")

    def set_corners(self, corners):
        g = self.calib.setdefault("geometry", {})
        g.update({"corners": [[float(x), float(y)] for x, y in corners], "source": "manual",
                  "time": time.time()})
        g.setdefault("lens", {"k": 0.0})
        self.save_calib()
        self._build_overview()
        self.say("Pool corners set by hand.")

    def add_ignore(self, rect):
        g = self.calib.setdefault("geometry", {})
        g.setdefault("ignore", []).append([float(v) for v in rect])
        self.save_calib()
        self._build_overview()
        self.say("Ignore zone added (motion there is not a swimmer).")

    def clear_ignore(self):
        g = self.calib.setdefault("geometry", {})
        g["ignore"] = []
        self.save_calib()
        self._build_overview()

    # ================================================================ commands
    def command(self, action, arg=None):
        with self.lock:
            if action == "track":
                self.force_track = {"on": True, "off": False, "auto": None}.get(arg, None)
                if self.force_track is False:
                    self.motion.stop()
                self.say({True: "Tracking forced ON.", False: "Tracking OFF.",
                          None: "Tracking back to automatic (races only)."}[self.force_track])
            elif action == "toggle":
                on = not self._tracking_effective()
                self.force_track = on
                if not on:
                    self.motion.stop()
                self.say("Tracking ON." if on else "Tracking OFF.")
            elif action == "leader":
                self.mode, self.lane = "leader", None
                self.say("Following the race leader.")
            elif action == "lane":
                self.mode, self.lane = "lane", int(arg)
                self.say(f"Locked onto lane {int(arg)}.")
            elif action == "zoom":
                step = 0.12
                if arg == "tighter":
                    self.zoom_bias -= step
                elif arg == "wider":
                    self.zoom_bias += step
                elif arg == "reset":
                    self.zoom_bias = 0.0
            elif action == "race_reset":
                self._set_phase("idle", "reset by operator")
                if self.ov:
                    self.ov.reset_race()
            elif action == "autocal":
                if arg == "stop":
                    self._autocal_end("stopped by operator")
                else:
                    self._autocal_start()
            elif action == "geometry":
                self.start_auto_geometry()
            elif action == "ptz_reset":
                self.model.reset()
                self.say("PTZ learning cleared.")
            elif action == "teach_undo":
                self.model.undo_teach()
                self.model.maybe_refit(force=True)
            else:
                raise ValueError(f"unknown action {action}")

    def teach(self, nx, ny):
        """Operator: 'the PTZ is framing this spot' (click on the overview)."""
        if not self.ov:
            return "pool geometry is not set yet"
        pos = self.ptz.position(1.5) if self.ptz else None
        if not pos:
            return "no PTZ position (check camera IP/login)"
        w, h = self.ov.size
        x, y = self.ov.img_to_pool(nx * w, ny * h)
        self.model.add_teach(x, y, pos[0], pos[1])
        self.model.maybe_refit(force=True)
        self.say(f"Taught: PTZ framing {x:.1f} m, lane {self.ov.lane_number(int(y // self.ov.lw))} "
                 f"(pan {pos[0]:04X} tilt {pos[1]:04X}).")
        return None

    def click_lock(self, nx, ny):
        if not self.ov:
            return
        w, h = self.ov.size
        x, y = self.ov.img_to_pool(nx * w, ny * h)
        if 0 <= y < self.ov.lanes * self.ov.lw and -1 <= x <= self.ov.L + 1:
            self.command("lane", self.ov.lane_number(int(y // self.ov.lw)))

    # ================================================================ main loop
    def _loop(self):
        seq_o = seq_p = 0
        last_t = time.monotonic()
        last_render = 0.0
        last_vars = 0.0
        while not self._stop:
            try:
                frame, s, _ = self.src["overview"].get(seq_o, timeout=0.25)
                now = time.monotonic()
                dt = clamp(now - last_t, 1e-3, 0.5)
                pframe, ps, pt = self.src["ptz"].get(seq_p)
                new_p = pframe is not None and ps != seq_p
                if new_p:
                    seq_p = ps
                    self._pframe_t = pt - float(self.cfg.get("ptz_feed", "video_delay_s", default=0.1))
                with self.lock:
                    if frame is not None and s != seq_o:
                        seq_o = s
                        last_t = now
                        self._step(frame, pframe if new_p else None, now, dt)
                    elif new_p:
                        self._ptz_step(pframe, now)
                    else:
                        self._idle_step(now)
                    if now - last_vars > 0.5:
                        self._push_vars()
                        last_vars = now
                    if now - last_render > 0.12 and self._want_stream():
                        self._render(frame if frame is not None else self.src["overview"].get(0)[0],
                                     pframe if pframe is not None else self.src["ptz"].get(0)[0], now)
                        last_render = now
                self.model.maybe_refit()
            except Exception as e:
                log.exception("engine loop")
                self.say(f"Engine error: {e}", "warn")
                time.sleep(0.5)

    def _want_stream(self):
        t = time.monotonic()
        return any(t - v < 3.0 for v in self.stream_clients.values())

    def _idle_step(self, now):
        return

    def _ptz_step(self, pframe, now):
        if pframe is None or self.pview is None:
            return
        self.pview.k_pan = self.learned.get("k_pan")
        self.pview.process(pframe)
        pos = self.ptz.position() if self.ptz else None
        z = self.ptz.zoom_value() if self.ptz else None
        b = self.pview.learn_fov(pos, z)
        if b is not None:
            self.learned["fov"] = {str(k): round(v, 1) for k, v in self.pview.fov.items()}
        self._learn_signs(now, pos)

    def _step(self, frame, pframe, now, dt):
        if self.geo_job is not None:
            self._geo_step(frame, now)
        if self.ov is None and self.geo_job is None and not self.calib.get("geometry") \
                and not getattr(self, "_geo_tried", False):
            self._geo_tried = True
            self.start_auto_geometry()
        if pframe is not None:
            self._ptz_step(pframe, now)
        if self.ov is None:
            return
        self.ov.process(frame, now)
        self._director(now)
        pos = self.ptz.position() if self.ptz else None
        if pos and (not self._pos_hist or self._pos_hist[-1][0] != pos[2]):
            self._pos_hist.append((pos[2], pos[0], pos[1]))
        self._detect_override(now, pos)
        self._learn(now, pos)
        if self.autocal:
            self._autocal_step(now, dt, pos)
        else:
            self._control(now, dt, pos)

    # ================================================================ race understanding
    def _active_lanes(self, now, fresh=0.5):
        return [s for s in self.ov.state.values() if now - s.mt < fresh and s.hits >= 3]

    def _set_phase(self, ph, why):
        if ph == self.phase:
            return
        self.phase, self.phase_t = ph, time.time()
        self.say(f"Phase → {ph.upper()}: {why}")

    def _director(self, now):
        c = self.cfg.snapshot()["director"]
        L = self.ov.L
        act = self._active_lanes(now)
        moving = [s for s in act if abs(s.v) > 0.4]
        quiet = len(moving) <= 1
        if quiet:
            self.quiet_since = self.quiet_since or now
        else:
            self.quiet_since = None

        if self.phase in ("idle", "warmup", "ready"):
            # quiet water -> watch for a start (a start can come from an empty pool, too)
            if self.quiet_since and now - self.quiet_since > c["start_quiet_s"] and self.phase != "ready":
                self._set_phase("ready", f"water quiet for {c['start_quiet_s']:.0f} s - watching for a start")
                self._start_watch = {}
            elif not quiet and self.phase == "idle":
                self._set_phase("warmup", f"{len(moving)} swimmers moving, no start seen")
            if self.phase == "ready":
                sw = self._start_watch
                for s in act:
                    near = 0 if s.x < 9 else (1 if s.x > L - 9 else None)
                    if near is not None and s.idx not in sw:
                        sw[s.idx] = (now, near)
                for k in [k for k, (t, _) in sw.items() if now - t > 3.0]:
                    del sw[k]
                for wall in (0, 1):
                    lanes = [k for k, (t, w) in sw.items() if w == wall]
                    if len(lanes) >= int(c["start_min_lanes"]):
                        span = max(sw[k][0] for k in lanes) - min(sw[k][0] for k in lanes)
                        self._race_start(now, wall, lanes, span)
                        return
                if len(moving) >= 2 and not sw:
                    self._set_phase("warmup", f"{len(moving)} swimmers moving without a start")
        elif self.phase == "race":
            r = self.race
            ld = self.ov.leader(now)
            if ld is not None:
                if self.last_leader is not None and ld.idx != self.last_leader:
                    self.say(f"Leader L{self.ov.lane_number(self.last_leader)} → L{self.ov.lane_number(ld.idx)}.")
                    if now - r.get("lead_evt", 0) > 3:
                        self.event("leader_change", f"L{self.ov.lane_number(ld.idx)}")
                        r["lead_evt"] = now
                self.last_leader = ld.idx
                if ld.laps > r.get("laps_seen", 0):
                    r["laps_seen"] = ld.laps
                    self.say(f"L{self.ov.lane_number(ld.idx)} turned - now on length {ld.laps + 1}.")
            if not r.get("live_sent") and now - r["t"] > c["live_delay_s"]:
                self.event("race_live", "race under way")
                r["live_sent"] = True
            # finish: the leader reached a wall and stopped, after at least one length
            if ld is not None and self.ov.progress(ld) and self.ov.progress(ld) > 0.8 * L:
                at_wall = ld.x < 1.5 or ld.x > L - 1.5
                stopped = now - ld.mt > 1.2 or abs(ld.v) < 0.25
                exp = r.get("expected_laps")
                plausible = exp is None or ld.laps + 1 >= exp
                if at_wall and stopped and plausible and ld.pred_turn_t <= ld.mt:
                    r.setdefault("touch_t", now)
                    # a turn resurfaces within ~2.5 s; wait that long unless the distance
                    # matches what recent heats swam
                    hold = 1.0 if exp and ld.laps + 1 == exp else 2.8
                    if now - r["touch_t"] > hold:
                        self._race_finish(now, ld)
                        return
                else:
                    r.pop("touch_t", None)
            if not act and now - r["t"] > 20 and now - max((s.mt for s in self.ov.state.values()), default=0) > 10:
                self._set_phase("idle", "everyone left the water without a clear finish")
                self.race = None
        elif self.phase == "finish":
            if time.time() - self.phase_t > c["finish_hold_s"]:
                self.event("idle", "after finish")
                self._set_phase("idle", "finish hold over - waiting for the next heat")
                self.race = None
                self.ov.reset_race()

    def _race_start(self, now, wall, lanes, span):
        self.ov.reset_race()
        self.race = {"t": now, "wall": wall, "lanes": len(lanes), "wall_t": time.time(),
                     "expected_laps": self._expected_laps()}
        self.last_leader = None
        exp = self.race["expected_laps"]
        self._set_phase("race", f"{len(lanes)} lanes left the {'start' if wall == 0 else 'far'} wall "
                                f"within {span:.1f} s" + (f"; expecting {exp} lengths like recent races" if exp else ""))
        self.event("race_start", f"{len(lanes)} lanes")

    def _race_finish(self, now, ld):
        r = self.race
        dur = now - r["t"]
        laps = ld.laps + 1
        self.say(f"Finish: L{self.ov.lane_number(ld.idx)} touched first after {laps} length(s), {dur:.1f} s.")
        self.event("finish", f"L{self.ov.lane_number(ld.idx)}")
        self._set_phase("finish", "leader touched")
        try:
            with open(os.path.join(DATA_DIR, "races.jsonl"), "a") as f:
                f.write(json.dumps({"t": time.time(), "lengths": laps, "duration_s": round(dur, 1),
                                    "lanes": r["lanes"], "winner_lane": self.ov.lane_number(ld.idx)}) + "\n")
        except OSError:
            pass

    def _expected_laps(self):
        """Pattern from history: the most common race distance of the last few heats
        (heats of an event are usually the same distance)."""
        try:
            with open(os.path.join(DATA_DIR, "races.jsonl")) as f:
                rows = [json.loads(l) for l in f.readlines()[-6:]]
        except OSError:
            return None
        if len(rows) < 2:
            return None
        last = [r["lengths"] for r in rows[-3:]]
        if len(set(last)) == 1:
            return last[0]
        return None

    # ================================================================ targets & control
    def _tracking_effective(self):
        if self.force_track is not None:
            return self.force_track
        return bool(self.cfg.get("director", "auto")) and self.phase in ("race", "finish")

    def _target(self, now):
        if self.ov is None:
            return None
        if self.mode == "lane" and self.lane:
            return self.ov.lane(self.lane, now)
        if self.phase == "finish" and self.race is not None and self.last_leader is not None:
            return self.ov.state.get(self.last_leader)
        return self.ov.leader(now)

    def _rate(self, axis):
        return self.learned.get(f"rate_{axis}", 120.0)

    def _control(self, now, dt, pos):
        c = self.cfg.snapshot()
        tgt = self._target(now)
        self.aim, self.ptz_lock = None, False
        self.target_desc = (f"L{self.ov.lane_number(tgt.idx)} {tgt.x:.1f} m, length {tgt.laps + 1}"
                            if tgt else None)
        if tgt is not None:
            lead = c["director"]["lead_m"] * (tgt.dir or 0)
            self.aim = (clamp(tgt.x + lead, 0, self.ov.L), self.ov.lane_center_m(tgt.idx))
        active = self._tracking_effective() and now >= self.override_until
        if not active:
            if getattr(self, "_was_active", False):
                self.motion.stop()
                self.pan_ctl.reset()
                self.tilt_ctl.reset()
            self._was_active = False
            return
        if not self.model.trusted():
            if not getattr(self, "_warned_model", False):
                self.say("Can't steer yet: the PTZ hasn't been calibrated. Run Auto-calibrate during "
                         "warm-ups, or teach a few spots.", "warn")
                self._warned_model = True
            return
        if pos is None:
            self.motion.stop()
            return
        if not getattr(self, "_was_active", False):
            self.pan_ctl.reset()
            self.tilt_ctl.reset()
        self._was_active = True
        if self.aim is None:
            ps = self.pan_ctl.coast(dt)
            ts = self.tilt_ctl.coast(dt)
            self.motion.pan_tilt(self._cmd_sign("pan") * ps, self._cmd_sign("tilt") * ts)
            return
        pr = self.model.predict(*self.aim)
        if pr is None:
            return
        tp, tt, dist = pr
        rate_p, rate_t = self._rate("pan"), self._rate("tilt")
        lat = c["control"]["latency_s"] + (time.monotonic() - pos[2])
        pan_now = pos[0] + self._cmd_sign("pan") * self.pan_ctl.out * rate_p * lat
        tilt_now = pos[1] + self._cmd_sign("tilt") * self.tilt_ctl.out * rate_t * lat
        zoom = self.ptz.zoom_value()
        # the PTZ's own view: find the target swimmer where the model says it should be
        # in the picture, and correct by where it actually is
        pv = self.pview
        if pv is not None and pv.blobs and self._signs_known():
            kp, kt = self.learned["k_pan"], self.learned["k_tilt"]
            sw_p, sw_t, _ = self.model.predict(tgt.x, self.aim[1])
            fu = pv.fov_units(zoom)
            aspect = (pv.size[1] / pv.size[0]) if pv.size else 0.5625
            ex_exp = (sw_p - pan_now) / (kp * fu)
            ey_exp = (sw_t - tilt_now) / (kt * fu * aspect)
            best = None
            for ex, ey, a in pv.blobs:
                d = math.hypot(ex - ex_exp, (ey - ey_exp) * aspect)
                if d < 0.12 and (best is None or d < best[0]):
                    best = (d, ex, ey)
            if best is not None:
                dp, dtl = pv.offset_to_units(best[1], best[2], zoom, kp, kt)
                corr_p = clamp(pan_now + dp - sw_p, -3 * self.model.upd, 3 * self.model.upd)
                corr_t = clamp(tilt_now + dtl - sw_t, -2 * self.model.upd, 2 * self.model.upd)
                w = float(c["ptz_feed"].get("weight", 0.7))
                tp, tt = tp + w * corr_p, tt + w * corr_t
                self.ptz_lock = True
                pv.last = (best[1], best[2])
        # feedforward from the swimmer's own speed
        p2 = self.model.predict(clamp(self.aim[0] + tgt.v * 0.25, 0, self.ov.L), self.aim[1])
        ffp = (p2[0] - pr[0]) / 0.25 / rate_p if p2 else 0.0
        fft = (p2[1] - pr[1]) / 0.25 / rate_t if p2 else 0.0
        sp = self.pan_ctl.update((tp - pan_now) / rate_p, dt, ffp)
        st = self.tilt_ctl.update((tt - tilt_now) / rate_t, dt, fft)
        self._send_pt(self._cmd_sign("pan") * sp, self._cmd_sign("tilt") * st)
        if c["director"]["auto_zoom"]:
            self._auto_zoom(dist, now)

    def _send_pt(self, p, t):
        self.motion.pan_tilt(p, t)
        self._cmd_hist.append((time.monotonic(), self.motion.last[0], self.motion.last[1]))

    def _frame_width(self):
        return float(self.learned.get("frame_width_m", self.cfg.get("director", "frame_width_m")))

    def _zoom_for(self, dist, width_m):
        want_deg = math.degrees(2 * math.atan(width_m / 2 / max(dist, 0.5)))
        want_units = want_deg * self.model.upd
        best = None
        for z in range(ZOOM_MIN, ZOOM_MAX + 1, 32):
            e = abs(math.log(self.pview.fov_units(z) / want_units))
            if best is None or e < best[0]:
                best = (e, z)
        return best[1]

    def _auto_zoom(self, dist, now):
        width = self._frame_width() * math.exp(self.zoom_bias)
        z = self._zoom_for(dist, width)
        lastz, lt = self._last_zoom_cmd
        if (lastz is None or abs(z - lastz) >= 40) and now - lt > 0.5:
            self.motion.zoom(z)
            self._last_zoom_cmd = (z, now)

    # ================================================================ learning
    def _cmd_sign(self, axis):
        return self.learned.get(f"c_{axis}", 1)

    def _signs_known(self):
        return "k_pan" in self.learned and "k_tilt" in self.learned

    def _learn_signs(self, now, pos):
        """From normal operation: which way the picture moves when pan/tilt values change,
        which way a speed command moves the values, and how fast."""
        if pos is None:
            return
        if not self._pos_hist or self._pos_hist[-1][0] != pos[2]:
            self._pos_hist.append((pos[2], pos[0], pos[1]))
        if len(self._pos_hist) < 3:
            return
        (t0, p0, q0), (t1, p1, q1) = self._pos_hist[-3], self._pos_hist[-1]
        if t1 - t0 < 0.2:
            return
        vp, vt = (p1 - p0) / (t1 - t0), (q1 - q0) / (t1 - t0)
        pv = self.pview
        if pv.shift_ok:
            dx, dy = pv.shift_px
            for key, v, d in (("k_pan", vp, dx), ("k_tilt", vt, dy)):
                if abs(v) > 150 and abs(d) > 1.0:
                    self._sign_ev[key] += math.copysign(1, v) * math.copysign(1, -d)
                    if abs(self._sign_ev[key]) >= 6 and key not in self.learned:
                        self.learned[key] = int(math.copysign(1, self._sign_ev[key]))
                        self.say(f"Learned how the PTZ picture moves with {key[2:]} values.")
                        self.save_calib()
        # command -> motion (only while we are the ones moving it, under a steady command)
        cmd = [h for h in self._cmd_hist if t0 - 0.6 <= h[0] <= t1]
        if len(cmd) >= 3 and now >= self.override_until:
            for axis, col, v in (("pan", 1, vp), ("tilt", 2, vt)):
                vals = {h[col] for h in cmd}
                if len(vals) != 1:
                    continue
                r = vals.pop()
                if abs(r) < 3 or abs(v) < 100:
                    continue
                self._sign_ev["c_" + axis] += math.copysign(1, r) * math.copysign(1, v)
                ev = self._sign_ev["c_" + axis]
                if abs(ev) >= 4 and int(math.copysign(1, ev)) != self.learned.get("c_" + axis):
                    self.learned["c_" + axis] = int(math.copysign(1, ev))
                    self.say(f"Learned the {axis} speed direction of this camera.")
                    self.save_calib()
                samples = self.__dict__.setdefault("_rate_samples", {}).setdefault(axis, [])
                samples.append(abs(v / r))
                del samples[:-25]
                if len(samples) >= 3:
                    self.learned["rate_" + axis] = round(float(np.median(samples)), 1)

    def _detect_override(self, now, pos):
        """Someone else (RP joystick, Companion) is moving the camera: it moves while we
        ask it to hold still, moves the other way, or much faster than we asked."""
        if pos is None or len(self._pos_hist) < 3:
            return
        if self.autocal and self.autocal.get("stage", "").startswith("dir"):
            return                                   # we're measuring directions right now
        if now < getattr(self, "_ignore_motion_until", 0):
            return
        (t0, p0, q0), (t1, p1, q1) = self._pos_hist[-3], self._pos_hist[-1]
        if t1 - t0 < 0.2 or now - t1 > 1.0:
            return
        vp, vt = (p1 - p0) / (t1 - t0), (q1 - q0) / (t1 - t0)
        # any command we sent shortly before/during this window may explain the motion
        win = [h for h in self._cmd_hist if t0 - 0.9 <= h[0] <= t1]
        before = [h for h in self._cmd_hist if h[0] < t0 - 0.9]
        if before:
            win = before[-1:] + win
        mine = bool(self._tracking_effective() and now >= self.override_until) or bool(self.autocal)
        why = []

        def odd(v, axis, col):
            if abs(v) <= 350:
                return False
            same = [abs(h[col]) * self._rate(axis) for h in win
                    if h[col] and math.copysign(1, self._cmd_sign(axis) * h[col]) == math.copysign(1, v)]
            if not same:
                why.append(f"{axis} moving {v:.0f}/s, not asked to")
                return True
            if abs(v) > 2.5 * max(same) + 600:
                why.append(f"{axis} moving {v:.0f}/s, faster than asked ({max(same):.0f}/s)")
                return True
            return False
        if mine:
            bad = odd(vp, "pan", 1) | odd(vt, "tilt", 2)
        else:
            bad = abs(vp) > 250 or abs(vt) > 250
        zs = self.ptz.zoom
        zl, zt = self._last_zoom_cmd
        if zs and getattr(self, "_zprev", None) is not None and abs(zs[0] - self._zprev) > 60 and now - zt > 3.0:
            bad = True
            why.append("zoom changed")
        self._zprev = zs[0] if zs else None
        self._ovr_hits = self._ovr_hits + 1 if bad else 0
        if self._ovr_hits >= 3:
            pause = self.cfg.get("director", "override_pause_s")
            if mine:
                self.say(f"You took the camera ({'; '.join(why) or 'unexpected motion'}) - hands off for {pause:.0f} s.")
                self.motion.stop()
                if self.autocal:
                    self._autocal_end("operator moved the camera")
            self.override_until = now + pause
            self._operator_seen = True
            self._op_hold = None
            self._ovr_hits = 0

    def _pos_at(self, t):
        """Camera pan/tilt at time t, interpolated from the polled positions."""
        h = list(self._pos_hist)
        if len(h) < 2:
            return None
        if t >= h[-1][0]:
            (ta, pa, qa), (tb, pb, qb) = h[-2], h[-1]
            if t - tb > 0.6 or tb <= ta:
                return None
            k = (t - tb) / (tb - ta)
            return pb + k * (pb - pa), qb + k * (qb - qa)
        for (ta, pa, qa), (tb, pb, qb) in zip(h, h[1:]):
            if ta <= t <= tb and tb > ta:
                k = (t - ta) / (tb - ta)
                return pa + k * (pb - pa), qa + k * (qb - qa)
        return None

    def _learn(self, now, pos):
        """Collect PTZ-view <-> overview evidence for the PTZ model; learn framing from
        the operator."""
        pv = self.pview
        every = 0.25 if self.autocal else 0.5
        if pv is None or pos is None or not self._signs_known() or now - self._sample_t < every:
            return
        self._sample_t = now
        cnt = self.__dict__.setdefault("learn_stats", {})
        zoom = self.ptz.zoom_value()
        if zoom is None or now - pos[2] > 0.6:
            cnt["no_pos"] = cnt.get("no_pos", 0) + 1
            return
        pt = getattr(self, "_pframe_t", None)
        at = self._pos_at(pt) if pt else None
        if at is None or now - pt > 0.5:
            cnt["no_sync"] = cnt.get("no_sync", 0) + 1
            return
        pos = (at[0], at[1], pt)
        # don't learn while whipping around (latency smears directions)
        if len(self._pos_hist) >= 2:
            (ta, pa, qa), (tb, pb, qb) = self._pos_hist[-2], self._pos_hist[-1]
            if tb > ta and math.hypot(pb - pa, qb - qa) / (tb - ta) > 12 * self.model.upd:
                cnt["fast"] = cnt.get("fast", 0) + 1
                return
        dirs = []
        for ex, ey, a in pv.blobs:
            if abs(ex) > 0.45 or abs(ey) > 0.45:
                continue
            dp, dt_ = pv.offset_to_units(ex, ey, zoom, self.learned["k_pan"], self.learned["k_tilt"])
            dirs.append((pos[0] + dp, pos[1] + dt_, ex, ey))
        # blob tracks in camera angles (same splash over time = same id)
        tracks = []
        offs = {}
        for p_, t_, ex, ey in dirs:
            best = None
            for tr in self._btracks:
                d = math.hypot(tr[1] - p_, tr[2] - t_)
                if d < 2.5 * self.model.upd and now - tr[3] < 1.2 and (best is None or d < best[0]):
                    best = (d, tr)
            if best:
                tid = best[1][0]
                self._btracks.remove(best[1])
            else:
                self._btid += 1
                tid = self._btid
            tracks.append((tid, p_, t_, now))
            offs[tid] = pv.tangent(ex, ey, zoom)
        self._btracks = tracks + [tr for tr in self._btracks if now - tr[3] < 1.2]
        swimmers = [s for s in self.ov.state.values() if now - s.mt < 0.3]
        if not tracks or not swimmers:
            cnt["no_blobs" if not tracks else "no_swimmers"] = cnt.get("no_blobs" if not tracks else "no_swimmers", 0) + 1
            return
        cnt["ok"] = cnt.get("ok", 0) + 1
        self.model.k = (self.learned["k_pan"], self.learned["k_tilt"])
        cands = [(s.x, self.ov.lane_center_m(s.idx), pos[0], pos[1], tid, s.idx, offs[tid][0], offs[tid][1])
                 for (tid, p_, t_, _) in tracks for s in swimmers]
        if self.model.trusted():
            pick = self.model.pick(cands)
            if pick is not None:
                self.model.add_frame([pick], weight=1.0)
        else:
            self.model.add_frame(cands, weight=1.0)
        # operator framing: when they hold a swimmer near centre, learn their zoom taste
        operating = now < self.override_until + 10 and getattr(self, "_operator_seen", False)
        if operating and pv.last is not None and abs(pv.last[0]) < 0.12 and self.model.trusted():
            r = self.model.ray_to_pool(pos[0], pos[1])
            if r is not None:
                pr = self.model.predict(*r)
                if pr:
                    width_now = 2 * pr[2] * math.tan(math.radians(pv.fov_units(zoom) / self.model.upd / 2))
                    h = self._op_hold
                    if h is None or abs(h["z"] - zoom) > 40:
                        self._op_hold = {"t": now, "z": zoom, "w": width_now}
                    elif now - h["t"] > 2.5 and not h.get("done"):
                        h["done"] = True
                        old = self._frame_width()
                        new = clamp(0.7 * old + 0.3 * width_now, 2.5, 25.0)
                        if abs(new - old) > 0.2:
                            self.learned["frame_width_m"] = round(new, 2)
                            self.say(f"Learned from you: framing about {new:.1f} m wide (was {old:.1f}).")
                            self.save_calib()

    # ================================================================ auto-calibration sweep
    def _autocal_start(self):
        pos = self.ptz.position(1.5) if self.ptz else None
        if pos is None:
            self.say("Auto-calibrate needs the PTZ position - check the camera IP/login.", "warn")
            return
        if self.ov is None:
            self.say("Auto-calibrate needs the pool geometry first.", "warn")
            return
        a = self.cfg.snapshot()["autocal"]
        upd = self.model.upd
        self.autocal = {"stage": "dir_pan", "t": time.monotonic(), "home": pos[:2],
                        "rows": [pos[1] + r * upd for r in a["tilt_rows_deg"]],
                        "span": a["pan_span_deg"] * upd, "row": 0, "dir": 1,
                        "speed": a["speed"], "zoom": a["zoom"], "start_pos": pos}
        self.override_until = 0.0
        self.say("Auto-calibrate: checking camera directions, then sweeping the pool. "
                 "Leave the joystick alone for a couple of minutes.")

    def _autocal_end(self, why):
        if not self.autocal:
            return
        home = self.autocal["home"]
        self.autocal = None
        self.motion.stop()
        if self.motion.mode == "direct":
            self.ptz.move_abs(*home)
            self._cmd_hist.append((time.monotonic(), 0, 0))
            self._ignore_motion_until = time.monotonic() + 4.0
        self.say(f"Auto-calibrate finished ({why}). Fitting the PTZ model…")
        self.model.maybe_refit(force=True)

    def _autocal_step(self, now, dt, pos):
        a = self.autocal
        el = now - a["t"]
        if pos is None:
            return
        if a["stage"] in ("dir_pan", "dir_tilt"):
            axis = a["stage"][4:]
            if "p0" not in a:
                a["p0"] = pos
                self._sign_ev["k_" + axis] = 0.0
            if el < 1.0:
                self._send_pt(15, 0) if axis == "pan" else self._send_pt(0, 15)
                return
            if el < 1.6:
                self._send_pt(0, 0)
                return
            dv = (pos[0] - a["p0"][0]) if axis == "pan" else (pos[1] - a["p0"][1])
            if abs(dv) < 50:
                self.say(f"The camera didn't report {axis} movement - is it controllable?", "warn")
                self._autocal_end("no movement")
                return
            self.learned["c_" + axis] = 1 if dv > 0 else -1
            self.learned["rate_" + axis] = round(abs(dv) / 1.0 / 15, 1)
            ev = self._sign_ev.get("k_" + axis, 0)
            if ev:
                self.learned["k_" + axis] = 1 if ev > 0 else -1
            self.save_calib()
            a.pop("p0")
            if axis == "pan":
                a["stage"], a["t"] = "dir_tilt", now
            else:
                self.motion.zoom(a["zoom"])
                self._last_zoom_cmd = (a["zoom"], now)
                a["stage"], a["t"] = "goto", now
                a["goal"] = (a["home"][0] - a["span"] / 2, a["rows"][0])
            return
        if a["stage"] == "goto":
            gp, gt = a["goal"]
            ep, et = gp - pos[0], gt - pos[1]
            if abs(ep) < 300 and abs(et) < 300 and el > 1.0:
                self._send_pt(0, 0)
                a["stage"], a["t"] = "sweep", now
                return
            sp = clamp(ep / self._rate("pan") * 0.6, -20, 20)
            st = clamp(et / self._rate("tilt") * 0.6, -12, 12)
            self._send_pt(self._cmd_sign("pan") * sp, self._cmd_sign("tilt") * st)
            if el > 25:
                self._autocal_end("could not reach the start of the sweep")
            return
        if a["stage"] == "sweep":
            end = a["home"][0] + a["dir"] * a["span"] / 2
            et = a["rows"][a["row"]] - pos[1]
            st = clamp(et / self._rate("tilt") * 0.5, -6, 6)
            self._send_pt(self._cmd_sign("pan") * a["dir"] * a["speed"], self._cmd_sign("tilt") * st)
            if (pos[0] - end) * a["dir"] >= 0 or el > 120:
                a["row"] += 1
                if a["row"] >= len(a["rows"]):
                    a["pass"] = a.get("pass", 1) + 1
                    if a["pass"] > a.get("max_passes", 3) or self.model.trusted():
                        self._autocal_end("sweep complete")
                        return
                    self.model.maybe_refit(force=True)
                    self.say(f"Sweep pass {a['pass'] - 1} done ({self.model.stats.get('state')}); sweeping again.")
                    a["row"] = 0
                a["dir"] = -a["dir"]
                a["stage"], a["t"] = "goto", now
                a["goal"] = (pos[0], a["rows"][a["row"]])

    # ================================================================ outputs
    def _push_vars(self):
        if not self.comp or not self.comp.enabled():
            return
        tgt = self.target_desc or ""
        ld = self.ov.leader(time.monotonic()) if self.ov else None
        self.comp.var("phase", self.phase)
        self.comp.var("tracking", 1 if self._tracking_effective() else 0)
        self.comp.var("leader", self.ov.lane_number(ld.idx) if ld else 0)
        self.comp.var("target", tgt)
        self.comp.var("mode", self.mode if self.mode == "leader" else f"lane {self.lane}")
        self.comp.var("lock", 1 if self.ptz_lock else 0)
        self.comp.var("override", 1 if time.monotonic() < self.override_until else 0)
        self.comp.var("length", (ld.laps + 1) if ld else 0)

    def state(self):
        now = time.monotonic()
        pos = self.ptz.position() if self.ptz else None
        z = self.ptz.zoom_value() if self.ptz else None
        ld = self.ov.leader(now) if self.ov else None
        swimmers = []
        if self.ov:
            for s in sorted(self.ov.state.values(), key=lambda s: s.idx):
                if self.ov._alive(s, now):
                    swimmers.append({"lane": self.ov.lane_number(s.idx), "x": round(s.x, 1),
                                     "v": round(s.v, 2), "length": s.laps + 1, "dir": s.dir,
                                     "leader": ld is not None and s.idx == ld.idx})
        g = self.calib.get("geometry") or {}
        return {
            "phase": self.phase, "phase_age": round(time.time() - self.phase_t, 1),
            "tracking": self._tracking_effective(), "force_track": self.force_track,
            "auto": bool(self.cfg.get("director", "auto")),
            "mode": self.mode, "lane": self.lane, "target": self.target_desc,
            "leader": self.ov.lane_number(ld.idx) if ld else None,
            "ptz_lock": self.ptz_lock, "override": max(0.0, round(self.override_until - now, 1)),
            "zoom_bias": round(self.zoom_bias, 2), "swimmers": swimmers,
            "camera": {"connected": bool(self.ptz and self.ptz.connected()),
                       "error": self.ptz.last_error if self.ptz else None,
                       "pos": [pos[0], pos[1]] if pos else None, "zoom": z,
                       "control": self.motion.mode if self.motion else None},
            "companion": {"enabled": bool(self.comp and self.comp.enabled()),
                          "connected": bool(self.comp and self.comp.connected()),
                          "error": self.comp.last_error if self.comp else None,
                          "log": [f"{time.strftime('%H:%M:%S', time.localtime(t))} {m}" for t, m in
                                  (self.comp.log[-8:] if self.comp else [])]},
            "feeds": {k: v.info() for k, v in self.src.items()},
            "ndi": {"ok": self.hub.ok, "error": self.hub.error, "sources": self.hub.names},
            "geometry": {"set": bool(g.get("corners")), "report": g.get("report"),
                         "quality": g.get("quality"), "source": g.get("source"),
                         "flip_lanes": g.get("flip_lanes", False), "flip_ends": g.get("flip_ends", False),
                         "ignore": len(g.get("ignore", [])), "busy": self.geo_job is not None},
            "ptz_model": dict(self.model.stats, teach=len(self.model.teach),
                              frames=len(self.model.frames), busy=self.model._busy),
            "learned": {k: v for k, v in self.learned.items() if k != "fov"} |
                       {"fov_bands": len(self.learned.get("fov", {}))},
            "autocal": (self.autocal or {}).get("stage"),
            "race": ({"elapsed": round(now - self.race["t"], 1),
                      "expected_lengths": self.race.get("expected_laps")} if self.race else None),
            "think": [{"t": time.strftime("%H:%M:%S", time.localtime(t)), "text": m, "level": lv}
                      for t, m, lv in list(self.think)[-40:]][::-1],
        }

    # ================================================================ overlays
    def _render(self, oframe, pframe, now):
        if oframe is not None:
            img = oframe.copy()
            if self.ov is not None and self.ov.size == (img.shape[1], img.shape[0]):
                self.ov.draw(img, now, self.aim)
                self._draw_footprint(img)
            elif self.ov is None:
                _label(img, "Pool not set - use Auto-detect pool in Calibration", (20, 40), (0, 200, 255), 0.8)
            self._draw_status(img, now)
            self.jpeg["overview"] = self._encode(img)
        if pframe is not None:
            img = pframe.copy()
            if self.pview is not None and self.pview.size:
                note = "LOCKED on target" if self.ptz_lock else (
                    "PTZ view: no swimmer near centre" if self._tracking_effective() else "PTZ view")
                self.pview.draw(img, self.ptz_lock, note)
            self.jpeg["ptz"] = self._encode(img)

    def _encode(self, img, width=960):
        h, w = img.shape[:2]
        if w > width:
            img = cv2.resize(img, (width, int(h * width / w)), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 72])
        return buf.tobytes() if ok else None

    def _draw_footprint(self, img):
        """Where the PTZ is looking, projected onto the overview."""
        pos = self.ptz.position() if self.ptz else None
        z = self.ptz.zoom_value() if self.ptz else None
        if pos is None or not self.model.params or self.pview is None:
            return
        fu = self.pview.fov_units(z)
        hu = fu * 0.5625
        kp, kt = self.learned.get("k_pan", 1), self.learned.get("k_tilt", 1)
        pts = []
        for ex, ey in ((-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)):
            r = self.model.ray_to_pool(pos[0] + kp * ex * fu, pos[1] + kt * ey * hu)
            if r is None:
                return
            pts.append(self.ov.pool_to_img(*r))
        col = (0, 255, 0) if self.ptz_lock else (0, 165, 255)
        cv2.polylines(img, [np.int32(pts)], True, col, 2)
        c = self.model.ray_to_pool(pos[0], pos[1])
        if c:
            cv2.drawMarker(img, tuple(map(int, self.ov.pool_to_img(*c))), col, cv2.MARKER_TILTED_CROSS, 14, 2)

    def _draw_status(self, img, now):
        lines = [f"{self.phase.upper()}  |  {'TRACKING' if self._tracking_effective() else 'not tracking'}"
                 f"{' (operator has camera)' if now < self.override_until else ''}"
                 f"  |  {'leader' if self.mode == 'leader' else f'lane {self.lane}'}",
                 f"target: {self.target_desc or '-'}   PTZ model: {self.model.stats.get('state')}"]
        if self.think:
            lines.append(self.think[-1][1][:110])
        h = img.shape[0]
        for i, t in enumerate(lines):
            _label(img, t, (12, h - 16 - 28 * (len(lines) - 1 - i)), (255, 255, 255), 0.6)
