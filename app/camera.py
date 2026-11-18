"""Panasonic AW PTZ over HTTP CGI: position/zoom polling + motion commands.

#PTSxxyy pan/tilt speed (50 = stop)   #APC -> aPCppppcccc   #GZ -> gzZZZ
#APCppppcccc absolute pan/tilt         #AXZzzz absolute zoom (555..FFF)
One worker thread serializes everything at >= min_interval: it round-robins position
polls, zoom polls and the latest pending pan/tilt and zoom commands.
"""
import logging
import re
import threading
import time

import requests

log = logging.getLogger("swim.camera")
ZOOM_MIN, ZOOM_MAX = 0x555, 0xFFF


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


class PanasonicPTZ:
    def __init__(self, host, user=None, password=None, min_interval=0.13):
        self.host = host
        self.url = f"http://{host}/cgi-bin/aw_ptz" if host else None
        self.s = requests.Session()
        if user:
            self.s.auth = requests.auth.HTTPDigestAuth(user, password or "")
        self.min_interval = min_interval
        self.send_motion = True          # False when motion goes through Companion
        self.pos = None                  # (pan, tilt, monotonic time)
        self.zoom = None                 # (zoom, monotonic time)
        self.last_ok = 0.0
        self.last_error = None
        self._pending, self._last = {}, {}
        self._cv = threading.Condition()
        self._lock = threading.Lock()
        self._t_send = 0.0
        self._slot = 0
        self._stop = False
        if self.url:
            threading.Thread(target=self._worker, daemon=True, name="ptz").start()

    def close(self):
        self._stop = True
        with self._cv:
            self._cv.notify_all()

    # ------------------------------------------------------------ transport
    def http(self, cmd, motion=False):
        if not self.url or (motion and not self.send_motion):
            return None
        with self._lock:
            wait = self.min_interval - (time.monotonic() - self._t_send)
            if wait > 0:
                time.sleep(wait)
            try:
                r = self.s.get(self.url, params={"cmd": cmd, "res": "1"}, timeout=0.6)
                self._t_send = time.monotonic()
                if r.status_code != 200:
                    self.last_error = f"HTTP {r.status_code}"
                    return None
                self.last_ok = time.monotonic()
                return r.text.strip()
            except requests.RequestException as e:
                self._t_send = time.monotonic()
                if self.last_error != e.__class__.__name__:
                    log.warning("camera %s unreachable: %s", self.host, e.__class__.__name__)
                self.last_error = e.__class__.__name__
                return None

    def query_position(self):
        m = re.search(r"aPC([0-9A-Fa-f]{4})([0-9A-Fa-f]{4})", self.http("#APC") or "")
        return (int(m.group(1), 16), int(m.group(2), 16)) if m else None

    def query_zoom(self):
        m = re.search(r"gz([0-9A-Fa-f]{3})", self.http("#GZ") or "")
        return int(m.group(1), 16) if m else None

    def _worker(self):
        """Alternate the newest pan/tilt command with position polls (~4 Hz each);
        zoom commands and zoom polls fill the gaps."""
        last = {"pts": 0.0, "pos": 0.0, "gz": 0.0}
        while not self._stop:
            now = time.monotonic()
            with self._cv:
                pick = None
                if "pts" in self._pending and now - last["pts"] >= 0.2:
                    pick = "pts"
                elif now - last["pos"] >= 0.22:
                    pick = "pos"
                elif "z" in self._pending:
                    pick = "z"
                elif now - last["gz"] >= 0.9:
                    pick = "gz"
                elif "pts" in self._pending:
                    pick = "pts"
                if pick is None:
                    self._cv.wait(timeout=0.03)
                    continue
                cmd = self._pending.pop(pick, None)
            last[pick if pick in last else "pts"] = time.monotonic() if pick != "z" else last["pts"]
            if pick == "pos":
                p = self.query_position()
                if p:
                    self.pos = (p[0], p[1], time.monotonic())
            elif pick == "gz":
                z = self.query_zoom()
                if z is not None:
                    self.zoom = (z, time.monotonic())
            elif cmd:
                self.http(cmd, motion=True)

    def _queue(self, ch, cmd):
        with self._cv:
            if self._last.get(ch) == cmd:
                return
            self._last[ch] = cmd
            self._pending[ch] = cmd
            self._cv.notify()

    # ------------------------------------------------------------ public
    def connected(self):
        return time.monotonic() - self.last_ok < 3.0

    def position(self, max_age=0.8):
        p = self.pos
        return p if p and time.monotonic() - p[2] <= max_age else None

    def zoom_value(self):
        return self.zoom[0] if self.zoom else None

    def set_pan_tilt(self, pan, tilt):
        p, t = int(round(clamp(pan, -49, 49))), int(round(clamp(tilt, -49, 49)))
        self._queue("pts", f"#PTS{50 + p:02d}{50 + t:02d}")

    def set_zoom_abs(self, z):
        self._queue("z", f"#AXZ{int(clamp(z, ZOOM_MIN, ZOOM_MAX)):03X}")

    def move_abs(self, pan, tilt):
        self.http(f"#APC{int(pan):04X}{int(tilt):04X}", motion=True)

    def stop(self):
        with self._cv:
            self._pending.pop("pts", None)
            self._last["pts"] = "#PTS5050"
        self.http("#PTS5050", motion=True)


class MotionOut:
    """Where pan/tilt/zoom commands go: straight to the camera, or into Companion custom
    variables (swim_pan / swim_tilt speeds 1..99, swim_zoom 0x555..0xFFF as decimal) for a
    Companion trigger to forward. Position is always read from the camera directly."""

    def __init__(self, ptz: PanasonicPTZ, companion, mode="direct"):
        self.ptz, self.comp, self.mode = ptz, companion, mode
        ptz.send_motion = mode == "direct"
        self.last = (0, 0)               # last commanded raw speeds (-49..49)

    def pan_tilt(self, p, t):
        p, t = int(round(clamp(p, -49, 49))), int(round(clamp(t, -49, 49)))
        self.last = (p, t)
        if self.mode == "companion":
            self.comp.var("pan", 50 + p)
            self.comp.var("tilt", 50 + t)
        else:
            self.ptz.set_pan_tilt(p, t)

    def zoom(self, z):
        z = int(clamp(z, ZOOM_MIN, ZOOM_MAX))
        if self.mode == "companion":
            self.comp.var("zoom", z)
        else:
            self.ptz.set_zoom_abs(z)

    def stop(self):
        self.last = (0, 0)
        if self.mode == "companion":
            self.comp.var("pan", 50)
            self.comp.var("tilt", 50)
        else:
            self.ptz.stop()
