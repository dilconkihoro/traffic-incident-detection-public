# Web dashboard (supplementary)

A browser-based demo layer on top of the CLI: log in, upload a CCTV video, watch it get processed
with live progress, and browse past runs with their before (original upload) and after (annotated,
tracked) video, incident log, and full track log. This is a supplementary showcase (decision D-25,
`docs/07_DECISIONS_RISKS.md`), not a replacement for the CLI. The CLI
(`python -m trafficwatch run <video>`) plus its `incidents.csv` and annotated video output remain
the official interface and the primary, evaluated deliverable (decision D-13). This dashboard calls
the same `trafficwatch` pipeline underneath; it does not duplicate or change any of its detection,
tracking, or incident logic.

**Each finished run's page shows, from the top:** the original video you uploaded next to the
annotated result (a box drawn around every tracked vehicle, a red banner when an incident fires),
both downloadable; the incidents that were flagged, in the order they happened, click one to jump
the video to that exact moment; and a "Tracked vehicles" section explaining and listing every
vehicle's position in every frame (`tracks.csv`), the raw data the incidents and the video overlay
are both built from, also downloadable.

**On processing speed:** this runs on a CPU, with no dedicated GPU. A real video was measured at
roughly 1.5 frames per second. The dashboard shows honest live progress (frames processed so far,
incidents as they are flagged, elapsed time), not a literal real-time 30fps feed. Uploads longer
than `TW_MAX_UPLOAD_FRAMES` (default 180 frames, about 7 seconds at 25fps, worst case under two
minutes to process) are rejected at upload time with a clear message, so an oversized video cannot
quietly tie up the machine for several minutes without warning; raise it in `.env` if you want to
demo a longer clip and are willing to wait.

**A run cannot take the machine down.** Processing is CPU-heavy, so several limits stack: only one
run executes at a time (`TW_MAX_CONCURRENT_RUNS`, default 1; extra runs wait as "queued" and start
by themselves), a run uses at most half the CPU cores (`TW_MAX_CPU_THREADS`) at lowered priority so
the rest of the desktop stays responsive, uploads are length-capped as above, and any queued or
running run can be stopped with the **Cancel run** button on its Processing page (the half-written
output is deleted and the next queued run starts). A run that stops advancing is flagged on that
page after 90 seconds. Memory matters too: each running pipeline holds about 1.5 GB, which is why
runs are serialised rather than run side by side.

## Tech stack

| Layer | Stack |
|---|---|
| Backend | FastAPI, SQLite (an embedded index file, not a database server), `bcrypt` password hashing, signed session cookies (`itsdangerous`), Server-Sent Events for live progress, a background thread per run (no Celery, no Redis, no message queue) |
| Frontend | Vite, React 19, TypeScript (strict mode), Tailwind CSS v4, React Router v7 |

## Layout

```
webapp/
  backend/
    main.py            FastAPI app, startup settings validation (fails closed on a missing/short secret)
    auth.py             password hashing/verification, session cookies, login rate limiting
    db.py               SQLite schema: one `runs` row per upload, an index only
    jobs.py              background job runner, wraps trafficwatch.pipeline.Pipeline exactly as the CLI does
    routers/            /api/login, /api/logout, /api/uploads, /api/runs/*
    tests/               backend test suite (pytest)
  frontend/
    src/pages/           Login, Upload, Processing, Dashboard, RunDetail
    src/lib/              typed API client, auth context, formatting helpers
```

Results are read from the same `reports/<run_id>/` files the CLI already writes (`incidents.csv`,
`tracks.csv`, `annotated.mp4`). SQLite only indexes them, plus the path to your original upload,
for the dashboard's run list and before/after view; it is never a second source of truth for
incident data.

## Setup, start to end

Python environment, venv, and `.env` should already be set up per `docs/08_SETUP_AND_PORTABILITY.md`
sections 2 to 7. From there:

**1. Install Node.js 20+ and ffmpeg** (once per machine), if not already installed: see the
prerequisites table in `docs/08_SETUP_AND_PORTABILITY.md` section 2. ffmpeg is optional for the
CLI but required here: `cv2.VideoWriter` writes a codec (mp4v) that plays fine in a desktop player
like VLC but that no mainstream browser can decode, and an arbitrary uploaded file is not
guaranteed to be browser-friendly either. Both the original upload and the annotated result are
re-encoded with ffmpeg (to H.264) before they can be viewed, so the before/after comparison always
plays. An upload or a run fails with a clear error if ffmpeg is missing, rather than silently
producing an unplayable video.

**2. Install the dashboard's Python packages and create an admin account.** From the repo root,
with the venv active (the CLI does not need these packages, so they are in their own file):

```bash
pip install -r requirements-webapp.txt
```

Then create the admin login:

```bash
python scripts/set_admin_password.py
```

Follow the prompts (username, password, confirm). It prints three lines; add them to your own
`.env` (never commit it, never share the values):

```
TW_ADMIN_USERNAME=...
TW_ADMIN_PASSWORD_HASH=...
TW_WEBAPP_SECRET_KEY=...
```

**3. Start the backend**, from the repo root, with the venv active:

```bash
uvicorn webapp.backend.main:app --reload --env-file .env --port 8000
```

`--env-file .env` loads the values from step 2 (and any of the optional overrides listed in
`.env.example`) into the process; the app refuses to start if the secret key or admin credentials
are missing or malformed, by design. Leave this running in its own terminal.

**4. Start the frontend**, in a second terminal, from the repo root:

```bash
cd webapp/frontend
npm install
npm run dev
```

Open the URL it prints (`http://localhost:5173`). The dev server proxies every `/api/*` request to
the backend on port 8000 automatically; no separate configuration is needed.

**5. Use it.** Log in with the username and password from step 2. Upload a video, watch it process
on the Processing page, then browse it and every past run from the Dashboard.

Both servers need to be running at the same time for the dashboard to work; this is a local demo
setup, not a deployed service.

## Sample clips to demo with

`data/samples/` has three short (170-frame, 6 to 20 second), real UA-DETRAC clips, each confirmed to
make the real detector and rules flag its incident type, so you don't have to hunt for footage
yourself:

| File | Flags | Note |
|---|---|---|
| `sample_abrupt_stop.mp4` | ABRUPT_STOP | Plays at 25fps |
| `sample_lane_obstruction.mp4` | LANE_OBSTRUCTION | Plays at about 8.3fps (a time-lapse): the rule needs 15s of stopped traffic, which needs more real time than a 170-frame demo clip can show at true speed |
| `sample_collision_like.mp4` | COLLISION_LIKE | The flagged moment is a car passing behind a stopped bus from the camera's angle, not a real crash; it is a genuine rule trigger, useful to demo the flow, but should not be described as footage of a real collision |
| `MVI_20011_short.mp4`, `MVI_20011.mp4` | none (by design) | Free-flowing traffic; good for showing a clean run with no false alarms |

`data/sample_search/kept.json` records exactly how each clip was made (source sequence, frame
range, ffmpeg command, checksums) if you ever need to regenerate or extend them.

## Production build

```bash
cd webapp/frontend
npm run build
```

Outputs a static bundle to `webapp/frontend/dist/` (git-ignored). Since this is a local
demonstration layer rather than a deployed web service (decision D-25 keeps deployment out of
scope, matching the CLI's own desktop-only scope guard), running the two dev servers together, as
in step 3 and 4 above, is the supported way to demo it; the production build exists to prove the
frontend compiles cleanly, and as a starting point should this ever need serving some other way.

## Backend tests

```bash
pytest webapp/backend/tests -q
```

52 tests: fast ones use a scripted (non-AI) detector to keep runs quick; one, marked `standin`, is
slower and drives the real stand-in model through the full HTTP API end to end.

## Good to know

- **Restarting the backend stops any run in progress.** Runs execute in threads inside the server
  process. On the next start, a run that was still queued or running is marked failed with the
  reason ("interrupted: the server was stopped or restarted..."); upload the video again to re-run
  it. Wait for a run to finish (or cancel it) before restarting, and use `--reload` only while
  developing, since every saved code change restarts the server and kills a run in progress.
- **Cancelling a stuck run.** If a run's page shows no progress (for example it belonged to a server
  that was restarted), **Cancel run** clears it: with nothing executing it, it is marked cancelled
  straight away.
- **Older runs still play.** Runs made before videos were normalised (and before the original
  upload's path was stored) are handled when you open them: the original is found through the run's
  own `run_manifest.json`, and a browser-playable copy (`*.web.mp4`) is made once beside the
  original file, which is left untouched.

## What this deliberately is not

- Not a replacement for the CLI, which stays the primary, tested, reviewed deliverable.
- Not a database server, message queue, or container deployment.
- Not literal real-time video processing (see the speed note above).
