"""Admin web UI + API (port 8080)."""
import asyncio
import logging
import os
import threading
import time

from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse

from .config import Config
from .director import Engine

logging.basicConfig(level=os.environ.get("SWIM_LOG", "INFO"),
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("swim.web")

app = FastAPI(title="SwimDirector")
STATIC = os.path.join(os.path.dirname(__file__), "static")
cfg = Config()
engine: Engine = None


@app.on_event("startup")
def _start():
    global engine
    engine = Engine(cfg)


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC, "index.html"), headers={"Cache-Control": "no-store"})


# ---------------------------------------------------------------- live views
@app.get("/stream/{which}.mjpg")
async def stream(which: str):
    if which not in ("overview", "ptz"):
        raise HTTPException(404)

    async def gen():
        last = None
        while True:
            engine.stream_clients[which] = time.monotonic()
            jpg = engine.jpeg.get(which)
            if jpg is not None and jpg is not last:
                last = jpg
                yield (b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " +
                       str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")
            await asyncio.sleep(0.08)
    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame",
                             headers={"Cache-Control": "no-store"})


@app.get("/snapshot/{which}.jpg")
def snapshot(which: str):
    engine.stream_clients[which] = time.monotonic()
    jpg = engine.jpeg.get(which)
    if jpg is None:
        raise HTTPException(404, "no picture yet")
    return Response(jpg, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


# ---------------------------------------------------------------- state & settings
@app.get("/api/state")
def state():
    return engine.state()


@app.get("/api/config")
def get_config():
    return cfg.public()


@app.post("/api/config")
def set_config(patch: dict = Body(...)):
    cam = patch.get("camera") or {}
    if cam.get("password") == "********":
        cam.pop("password")
    changed = cfg.update(patch)
    res = engine.configure(changed=changed)
    return {"ok": True, "changed": [".".join(k) for k in changed], "restart": res == "restart"}


@app.post("/api/restart")
def restart():
    # the container's restart policy brings us back with new NDI settings
    threading.Timer(0.5, lambda: os._exit(3)).start()
    return {"ok": True}


# ---------------------------------------------------------------- control
@app.post("/api/control")
def control(body: dict = Body(...)):
    try:
        engine.command(body.get("action"), body.get("arg"))
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"ok": True}


@app.post("/api/click")
def click(body: dict = Body(...)):
    """Clicks on the overview preview (normalized coords)."""
    kind, x, y = body.get("kind"), float(body["x"]), float(body["y"])
    if kind == "lock":
        engine.click_lock(x, y)
    elif kind == "teach":
        err = engine.teach(x, y)
        if err:
            raise HTTPException(409, err)
    else:
        raise HTTPException(400, "kind must be lock or teach")
    return {"ok": True}


@app.post("/api/geometry")
def geometry(body: dict = Body(...)):
    act = body.get("action")
    if act == "auto":
        engine.start_auto_geometry()
    elif act == "flip_lanes":
        engine.flip_geometry("lanes")
    elif act == "flip_ends":
        engine.flip_geometry("ends")
    elif act == "corners":
        pts = body.get("points") or []
        if len(pts) != 4:
            raise HTTPException(400, "need 4 corners")
        engine.set_corners(pts)
    elif act == "ignore":
        engine.add_ignore(body["rect"])
    elif act == "clear_ignore":
        engine.clear_ignore()
    else:
        raise HTTPException(400, "unknown geometry action")
    return {"ok": True}


# ---------------------------------------------------------------- Companion-friendly GETs
# Point a Companion "Generic HTTP" GET action at http://<host>:8080/cmd/...
@app.get("/cmd/{path:path}")
def cmd(path: str):
    p = [x for x in path.lower().split("/") if x]
    m = {"toggle": ("toggle", None), "on": ("track", "on"), "off": ("track", "off"),
         "auto": ("track", "auto"), "leader": ("leader", None), "tighter": ("zoom", "tighter"),
         "wider": ("zoom", "wider"), "zoomreset": ("zoom", "reset"), "reset": ("race_reset", None)}
    if p and p[0] == "lane" and len(p) == 2 and p[1].isdigit():
        engine.command("lane", int(p[1]))
    elif p and p[0] in m:
        engine.command(*m[p[0]])
    else:
        return JSONResponse({"error": "unknown command", "try": list(m) + ["lane/N"]}, 404)
    s = engine.state()
    return {k: s[k] for k in ("phase", "tracking", "mode", "lane", "leader", "target")}
