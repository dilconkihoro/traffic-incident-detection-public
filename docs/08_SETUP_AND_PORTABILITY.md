# 08. Setup, from a new machine to a running system

This guide takes a new laptop from nothing installed to a working pipeline run, on Windows,
Ubuntu, or macOS. Each step gives the command for each platform. Commands are shown in the
shell you should use on that platform.

**Verification status, read this first.** The steps were run end to end on Ubuntu Linux (x86_64,
CPU only, Python 3.10): clone, virtual environment, pinned install, the full test suite (536
tests passing), the CLI, and the dataset tools. Windows and macOS steps use the same commands, but
they have not yet been run on those systems. Known risk on macOS: the pinned PyTorch build is a
CPU wheel from PyTorch's own index, which does not carry macOS builds, so section 5 gives a
workaround. If a step fails on Windows or macOS, the troubleshooting table in section 13 is the
first place to look. Report anything not covered there.

## 1. What is in git and what is not

| In the repository | Not in the repository (recreate or fetch on each machine) |
|---|---|
| Source (`src/`), tests, scripts, configs, notebooks, documentation, `requirements*.txt`, `pyproject.toml`, `.env.example` | Python and Git themselves, the `.venv/` folder, your `.env`, the datasets, trained weights, run outputs |

Datasets and weights are excluded on purpose. Every raw archive and converted dataset is git-ignored,
so `git status` never lists them. Only hash and licence metadata (`data/MANIFEST.json`, a few
kilobytes) is tracked. GitHub also rejects any single file over 100 MB, so a large file cannot be
pushed by accident.

Expected disk space: about 12 GB for the raw archives, about 4 GB for the converted dataset, plus
a few GB for Python packages. Keep at least 25 GB free.

## 2. Install the prerequisites

Install these once per machine. Versions matter where stated.

### Windows (PowerShell, run as your normal user)

```powershell
winget install Python.Python.3.10
winget install Git.Git
winget install 7zip.7zip
winget install Gyan.FFmpeg
winget install OpenJS.NodeJS.LTS
```

- Python 3.10 is required (the project pins 3.10). If `winget` is not available, use the installer
  from python.org and tick **Add python.exe to PATH**.
- Close and reopen PowerShell after installing, so the new `PATH` entries load.
- The `ffmpeg` and `Node.js` lines are only needed for the optional web dashboard (section 12).
  `7zip` is needed to open the `.rar` dataset archive; Windows has no built-in support for it.
- Git for Windows also installs **Git Bash**. The demo script in section 10 needs Git Bash.

### Ubuntu or Debian (bash)

```bash
sudo apt update
sudo apt install -y git python3.10 python3.10-venv unrar ffmpeg nodejs npm
```

- If `python3.10` is not offered by your release, add the deadsnakes PPA first
  (`sudo add-apt-repository ppa:deadsnakes/ppa`), then install `python3.10` and `python3.10-venv`.
- `unrar` opens the `.rar` dataset archive. `ffmpeg` and `nodejs npm` are only needed for the web
  dashboard (section 12).

### macOS (Terminal, with Homebrew installed from brew.sh)

```bash
brew install git python@3.10 ffmpeg node unar
```

- `unar` opens the `.rar` dataset archive.
- Use the `python3.10` command that Homebrew provides for this formula.

### Check the versions

```powershell
# Windows, PowerShell
py -3.10 --version     # Python 3.10.x
git --version
ffmpeg -version | Select-Object -First 1   # only needed for section 12
```

```bash
# Ubuntu or macOS
python3.10 --version   # Python 3.10.x
git --version
ffmpeg -version | head -1   # only needed for section 12
```

## 3. Get the code

```bash
git clone https://github.com/dilconkihoro/traffic-incident-detection-public.git
cd traffic-incident-detection-public
```

The repository is public, so no account or token is needed to clone it. If you clone into a
folder with a different name, use that folder name in the commands below.

Set your name and email for commits in this clone only, if you plan to commit:

```bash
git config --local user.name "<your name>"
git config --local user.email "<your email>"
```

## 4. Create the virtual environment

Create one virtual environment inside the repository folder and activate it. Activate it again in
every new terminal before you run anything.

```powershell
# Windows, PowerShell
py -3.10 -m venv .venv
.venv\Scripts\Activate.ps1
```

If PowerShell refuses to run `Activate.ps1` with a script-execution error, allow scripts for the
current window only, then activate again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.venv\Scripts\Activate.ps1
```

```bash
# Ubuntu or macOS (also Git Bash on Windows, using .venv/Scripts/activate instead)
python3.10 -m venv .venv
source .venv/bin/activate
```

The prompt should now start with `(.venv)`.

## 5. Install the pinned packages

On **Windows and Ubuntu**, install the pinned requirements directly:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e . --no-deps
```

On **macOS**, the pinned line `torch==2.14.0+cpu` cannot be found, because PyTorch's CPU index
does not carry macOS builds. Install PyTorch from the default index first, then the rest of the
requirements without the two torch lines:

```bash
python -m pip install --upgrade pip
pip install torch==2.14.0 torchvision==0.29.0
grep -v -E "^(torch|torchvision)==|extra-index-url" requirements.txt > requirements-macos.txt
pip install -r requirements-macos.txt
pip install -e . --no-deps
rm requirements-macos.txt
```

If a step fails on macOS, send the error text; this step has not yet been run on a Mac.

The last command installs the project itself in editable mode, so `python -m trafficwatch` and
the scripts resolve to this checkout.

## 6. Create your `.env` file

`.env` holds your private settings. It is git-ignored; never commit it and never paste its
contents anywhere.

```powershell
# Windows, PowerShell
Copy-Item .env.example .env
```

```bash
# Ubuntu, macOS, or Git Bash
cp .env.example .env
```

Open `.env` in a text editor and set the values you need:

- Pipeline values work without any change; leave them unless you have a reason.
- Dataset mirror values (`KAGGLE_USERNAME`, `KAGGLE_KEY`, `HF_TOKEN`, `ROBOFLOW_API_KEY`) are only
  needed if you use those services. The DhakaAI download in section 8 needs none of them.
- Dashboard values (`TW_WEBAPP_SECRET_KEY`, `TW_ADMIN_USERNAME`, `TW_ADMIN_PASSWORD_HASH`) are
  only needed for section 12. The template explains how to generate each one.

The placeholder values in `.env.example` do not work as they are, so an unedited copy will not
let anyone log in.

## 7. Check that the install works

```bash
pytest -q -m "not standin"
ruff check . && ruff format --check .
python -c "import trafficwatch; print('import ok')"
```

Expected: the test suite passes (536 tests, about two minutes on a CPU laptop), lint is clean, and
the import prints `import ok`. The `standin` tests load the real stand-in model and are run in
section 10.

## 8. Get the datasets

Two datasets are needed. Both are free and both are public.

### 8a. DhakaAI (scripted, no account needed)

```bash
python scripts/acquire_datasets.py fetch-dhakaai --dry-run
python scripts/acquire_datasets.py fetch-dhakaai --retrieved <YYYY-MM-DD>
```

The first command lists the files and the licence and downloads nothing. The second downloads the
archives to `data/raw/dhakaai/` and records each file's SHA-256 hash in `data/MANIFEST.json`. Use
today's date for `<YYYY-MM-DD>`.

Extract the training archive so that the `Final Train Dataset` folder sits inside
`data/raw/dhakaai/train/`:

```powershell
# Windows, PowerShell (7-Zip)
New-Item -ItemType Directory -Force data\raw\dhakaai\train
7z x data\raw\dhakaai\train.rar -odata\raw\dhakaai\train
```

```bash
# Ubuntu
mkdir -p data/raw/dhakaai/train
unrar x data/raw/dhakaai/train.rar data/raw/dhakaai/train/
# macOS
mkdir -p data/raw/dhakaai/train && unar -o data/raw/dhakaai/train data/raw/dhakaai/train.rar
```

Confirm the folder name is exactly `Final Train Dataset`. The conversion script looks for it there.

### 8b. UA-DETRAC (manual download)

UA-DETRAC is not downloaded by a script. Its original host is no longer reliable, so obtain the
archives from a source you can access, using the source notes in `docs/03_DATA_PIPELINE.md`
section 9. You need three files:

- `DETRAC-Images.zip` (the video frames, about 10 GB)
- `DETRAC-Train-Annotations-XML.zip`
- `DETRAC-Test-Annotations-XML.zip`

Place them in `data/raw/ua-detrac/`, then unpack the two annotation archives in place, so that the
folders `DETRAC-Train-Annotations-XML/` and `DETRAC-Test-Annotations-XML/` sit inside
`data/raw/ua-detrac/`. Keep `DETRAC-Images.zip` as a zip file; the conversion reads it directly.

```powershell
# Windows, PowerShell
New-Item -ItemType Directory -Force data\raw\ua-detrac
Expand-Archive data\raw\ua-detrac\DETRAC-Train-Annotations-XML.zip -DestinationPath data\raw\ua-detrac
Expand-Archive data\raw\ua-detrac\DETRAC-Test-Annotations-XML.zip -DestinationPath data\raw\ua-detrac
```

```bash
# Ubuntu or macOS
mkdir -p data/raw/ua-detrac
unzip -q data/raw/ua-detrac/DETRAC-Train-Annotations-XML.zip -d data/raw/ua-detrac
unzip -q data/raw/ua-detrac/DETRAC-Test-Annotations-XML.zip -d data/raw/ua-detrac
```

Record each archive's hash and provenance once you have it:

```bash
python scripts/acquire_datasets.py verify data/raw/ua-detrac/DETRAC-Images.zip --dataset ua-detrac --retrieval manual --source-url "<where you got it>" --retrieved <YYYY-MM-DD>
```

Repeat the `verify` command for the two annotation zips. Then confirm the manifest has no blank fields:

```bash
python scripts/acquire_datasets.py check --strict
```

## 9. Build the training dataset (optional on a laptop)

Training itself runs on Colab (section 11). Building the converted dataset locally is only needed
if you want to rebuild the Colab upload, or check the conversion yourself:

```bash
python scripts/convert_to_yolo.py --dry-run     # reads and checks everything, writes nothing
python scripts/convert_to_yolo.py               # builds data/processed/yolo_ds_v1/ and its tar
```

The full build takes several minutes on a laptop. The output is `data/processed/yolo_ds_v1.tar`
and a matching `.sha256` file, which you upload to Google Drive for Colab.

## 10. Run the pipeline

### 10a. Download the stand-in model (once)

```bash
python scripts/fetch_standin_weights.py
pytest -m standin -q        # runs the real stand-in model on synthetic input
```

The stand-in is the public COCO-pretrained YOLOv8n model. Its output is always labelled
`coco_standin` in every file it writes, and its numbers never enter a results table.

### 10b. Run the command line on a video

```bash
python -m trafficwatch run <path-to-video.mp4> --standin
```

Every run writes `reports/<run_id>/` containing `incidents.csv`, `tracks.csv`, `annotated.mp4`,
`latency.json`, and `run_manifest.json`. Open `annotated.mp4` in any video player and
`incidents.csv` in any spreadsheet.

### 10c. The guided demo

The demo is a bash script that explains each step as it runs. It needs a 75-frame sample clip
at `data/samples/MVI_20011_short.mp4`, which you make once from the UA-DETRAC frames. Each frame
is a separate JPEG inside `DETRAC-Images.zip`, so extract the first 75 of sequence `MVI_20011`
and assemble them at the dataset's native 25 frames per second:

```bash
mkdir -p /tmp/mvi && unzip -q -j data/raw/ua-detrac/DETRAC-Images.zip "DETRAC-Images/MVI_20011/img000*.jpg" -d /tmp/mvi
mkdir -p data/samples
ffmpeg -framerate 25 -start_number 1 -i /tmp/mvi/img%05d.jpg -frames:v 75 -c:v libx264 -pix_fmt yuv420p data/samples/MVI_20011_short.mp4
```

On Windows, run these in Git Bash, or use the equivalent PowerShell commands and the same
ffmpeg arguments. Then run the demo:

```bash
bash scripts/run_full_demo.sh
```

On Windows, run the demo from **Git Bash**, not PowerShell. The script is bash, so PowerShell and
Command Prompt cannot run it.

## 11. Train on Colab

Training never runs on a laptop. The notebook needs a GPU, which Google Colab provides.

1. Upload `data/processed/yolo_ds_v1.tar` and `data/processed/yolo_ds_v1.sha256` to
   `MyDrive/tid/data/` in Google Drive. Upload with the Drive website, not Colab's own upload
   button, because the archive is several gigabytes.
2. In Colab, choose **File, Open notebook, GitHub**, and paste
   `https://github.com/dilconkihoro/traffic-incident-detection-public/blob/main/notebooks/colab_train.ipynb`.
   The repository is public, so no sign-in is needed to read it.
3. Set **Runtime, Change runtime type** to **T4 GPU**.
4. Run the cells from the top. The notebook clones the public repository itself, checks the
   dataset hash, runs a smoke test, then trains the runs, validates each on the validation split,
   and exports the results to Drive.

Each run's export is a zip in `MyDrive/tid/export/`. Copy it into `weights/<RUN_ID>/` in your
clone to use it locally. The notebook narrates each step, so the cells explain themselves.

## 12. The web dashboard (optional)

The dashboard is a supplementary browser layer on top of the same pipeline. The command line
in section 10 is the primary interface and works without it.

Setup needs Node.js and ffmpeg from section 2:

```bash
pip install -r requirements-webapp.txt
python scripts/set_admin_password.py      # prints the admin lines to paste into .env (see .env.example)
```

Generate the secret key and paste it into `.env`, using the command shown in `.env.example`, then
run both halves together. The run script copies `.env.example` to `.env` if no `.env` exists, so complete section 6 first, or
set the values right after the first run:

```bash
bash scripts/run_webapp_dev.sh           # Git Bash on Windows, or bash on Ubuntu and macOS
```

```powershell
# Windows PowerShell alternative
.\scripts\run_webapp_dev.ps1
```

Then open the address the script prints, log in with the admin account you set, and upload a short
video. Uploads over 180 frames (about 7 seconds at 25 fps) are rejected by default, so a long video
cannot tie up the laptop. Only one run executes at a time, and any queued or running run can be
stopped with its Cancel button. Full details are in `webapp/README.md`.

## 13. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `Activate.ps1 cannot be loaded because running scripts is disabled` | PowerShell default policy | `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`, then activate again (section 4) |
| `python` or `py` not found after install | Terminal opened before the install finished | Close and reopen the terminal; on Windows, reinstall with **Add python.exe to PATH** ticked |
| `bash: scripts/run_full_demo.sh: command not found` or PowerShell errors on `.sh` | Shell is not bash | Run `.sh` scripts from Git Bash (Windows) or a Terminal (macOS and Ubuntu) |
| `No matching distribution found for torch==2.14.0+cpu` on macOS | PyTorch CPU index has no macOS builds | Use the macOS install in section 5 |
| `unrar: command not found` / `.rar` will not open | Archive tool not installed | Ubuntu: `sudo apt install unrar`; macOS: `brew install unar`; Windows: install 7-Zip |
| `ModuleNotFoundError: No module named 'bcrypt'` | Dashboard packages not installed | `pip install -r requirements-webapp.txt` (section 12) |
| `No module named 'trafficwatch'` | Editable install skipped, or wrong environment active | Activate `.venv`, then run `pip install -e . --no-deps` |
| Path errors on Windows containing `data\processed\...` | Long-path limit | Enable long paths once as administrator: `New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force`, or keep the repository near the drive root |
| Git warns about line endings on Windows | Windows converts line endings on checkout | Harmless for this repository; `git config --global core.autocrlf true` silences it |
| `pip install` is very slow or times out | Large wheels over a slow connection | `pip install --default-timeout 120 --retries 5 -r requirements.txt` |
| `check --strict` lists unresolved fields | A manual archive was verified without its source URL or date | Re-run `verify` with `--source-url` and `--retrieved` |

## 14. What is the same on every platform

The project rules apply everywhere. Never commit `.env`, datasets, or weights. Stage files by name
(`git add <path>`), never with `git add -A` or `git add .`. Do not push unless you mean to. Do not
add a `Co-Authored-By` trailer to commits. Use no em dashes in documentation. None of these change
between Windows, Ubuntu, and macOS.
