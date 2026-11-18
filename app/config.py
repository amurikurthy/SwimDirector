"""Settings: defaults + /data/config.yaml (edited from the admin page)."""
import copy
import os
import threading

import yaml

DATA_DIR = os.environ.get("SWIM_DATA", "/data")

DEFAULTS = {
    "ndi": {
        "discovery_server": "",          # e.g. 10.100.100.20; empty = mDNS
    },
    "overview": {                        # static wide camera (GoPro)
        "type": "ndi",                   # ndi | url
        "name": "",                      # NDI source name (or part of it)
        "url": "",                       # rtsp://... or a video file, when type = url
    },
    "ptz_feed": {                        # the PTZ camera's own video
        "type": "ndi",
        "name": "",
        "url": "",
        "weight": 0.7,                   # how much the PTZ's own view refines the aim
        "video_delay_s": 0.1,            # PTZ picture lags its reported position by this much
    },
    "camera": {                          # Panasonic AW PTZ
        "host": "",
        "user": "admin",
        "password": "",
        "control": "direct",             # direct | companion   (how motion commands are sent)
        "units_per_deg": 121.4,
    },
    "companion": {
        "host": "",
        "port": 8000,
        "variable_prefix": "swim_",
        "buttons": {                     # "page/row/column" pressed on director events
            "race_start": "",
            "race_live": "",
            "leader_change": "",
            "finish": "",
            "idle": "",
        },
    },
    "pool": {
        "length_m": 25.0,                # 25-yard pool = 22.86
        "lanes": 8,
        "lane_width_m": 2.5,
        "lane_first": 1,
        "outer_ropes": "auto",           # auto | yes | no
    },
    "director": {
        "auto": True,                    # follow races automatically
        "lead_m": 1.0,                   # aim ahead of the swimmer
        "frame_width_m": 7.0,            # auto-zoom framing (learned from your corrections)
        "auto_zoom": True,
        "override_pause_s": 6.0,         # hands off this long after you touch the camera
        "finish_hold_s": 5.0,
        "start_quiet_s": 3.0,            # quiet water before a start
        "start_min_lanes": 3,
        "live_delay_s": 2.5,             # race_start -> race_live button
    },
    "control": {
        "latency_s": 0.12,
        "pan": {"kp": 2.0, "ki": 0.6, "deadband": 0.3, "max_speed": 45, "min_speed": 2,
                "slew": 150, "integral_limit": 10, "coast_seconds": 1.5},
        "tilt": {"kp": 1.6, "ki": 0.4, "deadband": 0.5, "max_speed": 30, "min_speed": 2,
                 "slew": 100, "integral_limit": 8, "coast_seconds": 1.5},
    },
    "autocal": {
        "pan_span_deg": 120.0,           # sweep width, centred on where the PTZ points at start
        "tilt_rows_deg": [-8.0, 0.0, 6.0],
        "speed": 6,                      # pan speed step while sweeping
        "zoom": 2048,                    # zoom held while sweeping (0x555 wide .. 0xFFF tele)
    },
    "vision": {
        "diff_threshold": 20,
        "min_mass": 25,
        "glare_rate": 0.35,
        "leader_margin_m": 0.7,
        "leader_hold_s": 0.6,
        "race_reset_s": 1e9,             # the director decides when races end
    },
}

# settings whose change needs a process restart (NDI reads them once)
RESTART_KEYS = {("ndi", "discovery_server")}


def deep_merge(base, over):
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def flat(d, prefix=()):
    for k, v in d.items():
        if isinstance(v, dict):
            yield from flat(v, prefix + (k,))
        else:
            yield prefix + (k,), v


class Config:
    def __init__(self, path=None):
        self.path = path or os.path.join(DATA_DIR, "config.yaml")
        self.lock = threading.Lock()
        self.data = copy.deepcopy(DEFAULTS)
        self.version = 0
        if os.path.exists(self.path):
            with open(self.path) as f:
                self.data = deep_merge(DEFAULTS, yaml.safe_load(f) or {})
        self._env_overrides()

    def _env_overrides(self):
        # handy for docker-compose: SWIM_CAMERA_HOST=10.0.0.5 etc. (first run only)
        for env, path in (("SWIM_NDI_DISCOVERY", ("ndi", "discovery_server")),
                          ("SWIM_OVERVIEW_NDI", ("overview", "name")),
                          ("SWIM_PTZ_NDI", ("ptz_feed", "name")),
                          ("SWIM_CAMERA_HOST", ("camera", "host")),
                          ("SWIM_COMPANION_HOST", ("companion", "host"))):
            v = os.environ.get(env)
            if v and not self.get(*path):
                self.data[path[0]][path[1]] = v

    def get(self, *path, default=None):
        d = self.data
        for p in path:
            if not isinstance(d, dict) or p not in d:
                return default
            d = d[p]
        return d

    def snapshot(self):
        with self.lock:
            return copy.deepcopy(self.data)

    def update(self, patch):
        """Merge a partial settings dict, save, and return the changed key paths."""
        with self.lock:
            before = dict(flat(self.data))
            self.data = deep_merge(self.data, patch)
            after = dict(flat(self.data))
            changed = {k for k in after if before.get(k) != after[k]}
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            tmp = self.path + ".tmp"
            with open(tmp, "w") as f:
                yaml.safe_dump(self.data, f, sort_keys=False)
            os.replace(tmp, self.path)
            self.version += 1
            return changed

    def public(self):
        """Settings for the admin page (password masked)."""
        d = self.snapshot()
        if d["camera"].get("password"):
            d["camera"]["password"] = "********"
        return d
