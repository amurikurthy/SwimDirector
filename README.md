# SwimDirector

A Docker container that watches your swim meet and points the PTZ for you.

- **Watches** the static overview camera (GoPro over NDI) and the PTZ's own NDI feed.
- **Understands the race**: finds the pool and lanes by itself, tracks every swimmer, counts
  lengths through turns, picks the leader, and spots starts and finishes.
- **Drives the PTZ** (Panasonic AW, HTTP) to frame the leader or a locked lane, with auto zoom.
- **Tells Companion** what's happening: presses your buttons on race start / live / leader change /
  finish / idle, and keeps status in custom variables for button feedback.
- **Learns**: the lens and pool layout, where the PTZ is mounted (from warm-ups or from watching
  you operate), how the camera responds, and how you like shots framed. The *Thinking* panel
  explains every decision.

## Run it

You need a machine with Docker. Save this as `docker-compose.yml` and start it:

```yaml
services:
  swimdirector:
    image: ghcr.io/amurikurthy/swimdirector:latest
    container_name: swimdirector
    ports:
      - "8080:8080"                # admin page and /cmd/... for Companion
    environment:
      TZ: "America/Chicago"
      # first-run defaults only; everything is editable in the admin page afterwards
      SWIM_NDI_DISCOVERY: "10.100.100.20"   # NDI Discovery Server; empty = mDNS
      SWIM_OVERVIEW_NDI: "NISZWBOX01"       # static overview camera (GoPro)
      SWIM_PTZ_NDI: "NISZWBOX02"            # the PTZ's own NDI feed
      SWIM_CAMERA_HOST: "10.215.230.202"    # Panasonic AW camera IP
      # SWIM_COMPANION_HOST: "10.0.0.20"    # Bitfocus Companion IP
    volumes:
      - ./data:/data               # settings, calibration, what it has learned: back this folder up
    restart: unless-stopped        # also used by "Save & restart" in the admin page
```

```bash
docker compose up -d
docker compose logs swimdirector      # startup and what it is doing
```

Open the admin page at `http://<docker-host>:8080`.

**Update:** `docker compose pull && docker compose up -d`. Your data in `./data` is kept.

The image is built for amd64 and arm64 by GitHub Actions on every push to `main`. To build from a
checkout of this repository instead, uncomment `build: .` in its `docker-compose.yml` and run
`docker compose up -d --build`.

### Settings

The `environment:` values are first-run defaults only: once `./data/config.yaml` exists, the admin
page is where you change things.

| Variable | Default | |
|---|---|---|
| `SWIM_NDI_DISCOVERY` | empty (mDNS) | NDI Discovery Server IP |
| `SWIM_OVERVIEW_NDI` | | NDI source name of the static overview camera |
| `SWIM_PTZ_NDI` | | NDI source name of the PTZ's own feed |
| `SWIM_CAMERA_HOST` | | Panasonic AW camera IP |
| `SWIM_COMPANION_HOST` | | Bitfocus Companion IP |
| `SWIM_LOG` | `INFO` | log level (`DEBUG` for more detail) |
| `SWIM_DATA` | `/data` | state directory inside the container |
| `TZ` | UTC | time zone for logs and race history |

Works with Docker Desktop on Windows/macOS and on Linux. With an NDI Discovery Server the normal
bridge network is fine; only plain mDNS discovery needs `network_mode: host` (Linux).

## First time at a pool (about 5 minutes, during warm-ups)

1. **Settings**: check the NDI sources (pick from the list), PTZ IP + login, Companion IP, and the
   pool (length: 25-yard pool = 22.86 m; lane count). Save.
2. **Calibration → Auto-detect pool** (runs by itself on first start too). The overview shows the
   lane grid it found; if lane 1 is on the wrong side or start/far are swapped, use the flip buttons.
   If it can't find the pool, click the 4 corners by hand.
3. **Ignore zones**: drag boxes over the backstroke flags and the deck edge where people stand.
4. **Teach the PTZ** (pick one, or both):
   - *Auto-calibrate (sweep)* during warm-ups: the PTZ checks its directions, then sweeps the pool
     while matching what it sees with the swimmers the overview tracks. Needs swimmers in the water
     and hands off the joystick for a couple of minutes (it repeats the sweep up to 3 times).
   - *Teach*: choose “Click = PTZ is framing this”, aim + zoom with your RP joystick, click that
     spot on the overview. 4–6 spread-out spots give a working model immediately.

   The calibration card shows where it thinks the PTZ is mounted (metres along / across / up).
   It won't steer until the fit is trustworthy, and it keeps refining in the background.

## During the meet

- **Auto (races)**: tracking turns on by itself at each start and off after the finish hold.
- **Track now / Off**: force it. **Leader** or a **lane number** chooses who to follow
  (or click a swimmer on the overview). **Wider / Tighter** biases the auto zoom.
- Grab the RP joystick any time: SwimDirector notices, backs off for a few seconds
  (`Hands-off after operator moves`), and learns from what you framed.

## Companion

Pushed custom variables (default prefix `swim_`): `phase`, `tracking`, `leader`, `target`, `mode`,
`lock`, `override`, `length`.

Buttons pressed on events (set `page/row/column` in Settings): race start, race live (after the
start delay), leader change, finish, back to idle.

Control SwimDirector from Companion with Generic HTTP GETs to `http://<host>:8080/cmd/...`:
`toggle`, `on`, `off`, `auto`, `leader`, `lane/4`, `wider`, `tighter`, `zoomreset`, `reset`.

PTZ moves go straight to the camera by default (fastest). Set *Send moves → Through Companion*
to have SwimDirector write `swim_pan` / `swim_tilt` (1–99, 50 = stop) and `swim_zoom`
instead, and build a Companion trigger that forwards them to your PTZ module.

## Files in `./data`

| File | What |
|---|---|
| `config.yaml` | settings from the admin page |
| `calib.json` | pool geometry, ignore zones, learned camera behaviour and framing |
| `ptzmodel.json` | PTZ mount model + the evidence it learned from |
| `races.jsonl` | race history (used to anticipate distances of the next heats) |

Delete `calib.json` / `ptzmodel.json` to start learning from scratch (or use the buttons).

## Tuning

- Phantom swimmers in empty lanes: add ignore zones, raise `vision.min_mass` in `config.yaml`.
- Missed swimmers: lower `vision.diff_threshold`.
- Starts not detected: lower *Lanes needed to call a start*; check *Quiet water before a start*.
- Camera lags / overshoots: `control.pan.kp` up / down.
- PTZ picture lags its reported position (smearing at speed): `ptz_feed.video_delay_s`.
