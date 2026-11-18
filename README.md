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

```bash
docker compose up -d --build
# admin page: http://<docker-host>:8080
```

`docker-compose.yml` sets first-run defaults (NDI Discovery Server, NISZWBOX01 / NISZWBOX02,
camera IP). Everything else lives in the admin page and is saved in `./data`.

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
