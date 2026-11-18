"""Bitfocus Companion HTTP API client (Companion 3+).

POST /api/location/<page>/<row>/<column>/press
POST /api/custom-variable/<name>/value?value=<value>
Sends from a background thread; variables are de-duplicated, the newest value wins.
"""
import logging
import queue
import threading
import time

import requests

log = logging.getLogger("swim.companion")


class Companion:
    def __init__(self, host="", port=8000, prefix="swim_"):
        self.base = f"http://{host}:{int(port or 8000)}" if host else None
        self.prefix = prefix or ""
        self.s = requests.Session()
        self._vars, self._sent = {}, {}
        self._presses = queue.Queue()
        self._cv = threading.Condition()
        self.last_ok = 0.0
        self.last_error = None
        self.log = []                     # recent (time, text) for the admin page
        if self.base:
            threading.Thread(target=self._worker, daemon=True, name="companion").start()

    def enabled(self):
        return self.base is not None

    def connected(self):
        return self.base is not None and time.monotonic() - self.last_ok < 15

    def var(self, name, value):
        if not self.base:
            return
        with self._cv:
            self._vars[self.prefix + name] = str(value)
            self._cv.notify()

    def press(self, location, why=""):
        """location 'page/row/column'."""
        if not self.base or not location:
            return
        self._presses.put((location.strip().strip("/"), why))
        with self._cv:
            self._cv.notify()

    def _post(self, path, params=None):
        try:
            r = self.s.post(self.base + path, params=params, timeout=1.0)
            if r.status_code < 300:
                self.last_ok = time.monotonic()
                self.last_error = None
                return True
            self.last_error = f"HTTP {r.status_code} on {path}"
        except requests.RequestException as e:
            self.last_error = e.__class__.__name__
        return False

    def _worker(self):
        while True:
            with self._cv:
                self._cv.wait(timeout=5.0)
                pending = {k: v for k, v in self._vars.items() if self._sent.get(k) != v}
            while not self._presses.empty():
                loc, why = self._presses.get()
                ok = self._post(f"/api/location/{loc}/press")
                self.log.append((time.time(), f"press {loc} ({why}) {'ok' if ok else 'FAILED'}"))
                self.log = self.log[-50:]
            for k, v in pending.items():
                if self._post(f"/api/custom-variable/{k}/value", {"value": v}):
                    self._sent[k] = v
            if not pending and self._presses.empty():
                # heartbeat so the admin page can show the link state
                if time.monotonic() - self.last_ok > 10:
                    self._post(f"/api/custom-variable/{self.prefix}heartbeat/value",
                               {"value": int(time.time())})
