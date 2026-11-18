"""Video inputs: NDI (with Discovery Server) or a URL/file, always keeping only the newest frame."""
import json
import logging
import os
import threading
import time

import numpy as np

from .config import DATA_DIR

log = logging.getLogger("swim.sources")


class NDIHub:
    """One NDI runtime per process + a finder that keeps the source list fresh."""

    def __init__(self, discovery_server=""):
        self.ok = False
        self.error = None
        self.sources = []           # NDI source objects
        self.names = []
        self.lock = threading.Lock()
        cfg_dir = os.path.join(DATA_DIR, "ndi")
        os.makedirs(cfg_dir, exist_ok=True)
        net = {"discovery": discovery_server} if discovery_server else {}
        with open(os.path.join(cfg_dir, "ndi-config.v1.json"), "w") as f:
            json.dump({"ndi": {"networks": net}}, f)
        os.environ["NDI_CONFIG_DIR"] = cfg_dir
        try:
            import NDIlib as ndi
            self.ndi = ndi
            if not ndi.initialize():
                raise RuntimeError("NDI runtime failed to initialize")
            self.finder = ndi.find_create_v2()
            self.ok = True
            threading.Thread(target=self._find_loop, daemon=True, name="ndi-find").start()
            log.info("NDI ready (discovery server: %s)", discovery_server or "mDNS")
        except Exception as e:  # no NDI: URL sources still work
            self.error = str(e)
            log.warning("NDI unavailable: %s", e)

    def _find_loop(self):
        while True:
            try:
                self.ndi.find_wait_for_sources(self.finder, 2000)
                srcs = self.ndi.find_get_current_sources(self.finder)
                with self.lock:
                    self.sources = list(srcs)
                    self.names = sorted(s.ndi_name for s in srcs)
            except Exception as e:
                log.warning("NDI find: %s", e)
                time.sleep(2)

    def match(self, name):
        if not name:
            return None
        with self.lock:
            exact = [s for s in self.sources if s.ndi_name == name]
            if exact:
                return exact[0]
            part = [s for s in self.sources if name.lower() in s.ndi_name.lower()]
            return part[0] if part else None


class VideoSource:
    """Reads one feed in the background. get() returns the newest frame (never a backlog)."""

    def __init__(self, label, cfg, hub: NDIHub):
        self.label = label
        self.cfg = dict(cfg)
        self.hub = hub
        self._frame = None
        self._seq = 0
        self._t = 0.0
        self._cond = threading.Condition()
        self.status = "starting"
        self.source_name = None
        self.fps = 0.0
        self.size = None
        self._stop = threading.Event()
        threading.Thread(target=self._run, daemon=True, name=f"src-{label}").start()

    # -------------------------------------------------------------- consumer side
    def get(self, last_seq, timeout=0.0):
        with self._cond:
            if timeout:
                self._cond.wait_for(lambda: self._seq != last_seq or self._stop.is_set(),
                                    timeout=timeout)
            return self._frame, self._seq, self._t

    def stop(self):
        self._stop.set()
        with self._cond:
            self._cond.notify_all()

    def info(self):
        live = time.monotonic() - self._t < 2.0 if self._t else False
        return {"status": "live" if live else self.status, "source": self.source_name,
                "fps": round(self.fps, 1), "size": self.size}

    # -------------------------------------------------------------- producer side
    def _put(self, frame):
        now = time.monotonic()
        with self._cond:
            if self._t:
                self.fps = 0.9 * self.fps + 0.1 / max(now - self._t, 1e-3)
            self._frame, self._t = frame, now
            self._seq += 1
            self.size = [int(frame.shape[1]), int(frame.shape[0])]
            self._cond.notify_all()

    def _run(self):
        if self.cfg.get("type", "ndi") == "url":
            self._run_url()
        else:
            self._run_ndi()

    def _run_url(self):
        import cv2
        url = self.cfg.get("url") or ""
        if not url:
            self.status = "not configured"
            return
        is_file = "://" not in url and not url.isdigit()
        while not self._stop.is_set():
            cap = cv2.VideoCapture(int(url) if url.isdigit() else url)
            if not cap.isOpened():
                self.status = "cannot open"
                time.sleep(3)
                continue
            self.status, self.source_name = "connected", url
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            while not self._stop.is_set():
                ok, frame = cap.read()
                if not ok:
                    if is_file:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    self.status = "lost"
                    break
                self._put(frame)
                if is_file:
                    time.sleep(1 / fps)
            cap.release()

    def _run_ndi(self):
        name = self.cfg.get("name") or ""
        if not name:
            self.status = "not configured"
            return
        if not self.hub.ok:
            self.status = f"NDI unavailable: {self.hub.error}"
            return
        ndi = self.hub.ndi
        while not self._stop.is_set():
            src = self.hub.match(name)
            if src is None:
                self.status = "searching"
                time.sleep(1)
                continue
            rc = ndi.RecvCreateV3()
            rc.color_format = ndi.RECV_COLOR_FORMAT_BGRX_BGRA
            if hasattr(ndi, "RECV_BANDWIDTH_LOWEST"):
                rc.bandwidth = ndi.RECV_BANDWIDTH_LOWEST   # preview stream: plenty, low CPU
            recv = ndi.recv_create_v3(rc)
            ndi.recv_connect(recv, src)
            self.status, self.source_name = "connecting", src.ndi_name
            last = time.monotonic()
            while not self._stop.is_set():
                t, v, a, m = ndi.recv_capture_v2(recv, 500)
                if t == ndi.FRAME_TYPE_VIDEO:
                    frame = np.copy(v.data[:, :, :3])
                    ndi.recv_free_video_v2(recv, v)
                    self._put(frame)
                    last = time.monotonic()
                elif t == ndi.FRAME_TYPE_AUDIO:
                    ndi.recv_free_audio_v2(recv, a)
                elif t == ndi.FRAME_TYPE_METADATA:
                    ndi.recv_free_metadata(recv, m)
                if time.monotonic() - last > 5:
                    self.status = "no video - reconnecting"
                    break
            ndi.recv_destroy(recv)
