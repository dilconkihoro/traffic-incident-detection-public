# 08. Setup, from a new machine to a running system

This guide takes a new laptop from nothing installed to a working pipeline run, on Windows,
Ubuntu, or macOS. Each step gives the command for each platform. Commands are shown in the
shell you should use on that platform.

**Verification status, read this first.** The steps were run end to end on Ubuntu Linux (x86_64,
CPU only, Python 3.10): clone, virtual environment, pinned install, the full test suite (536
tests passing), the CLI, and the dataset tools. The macOS steps were run on Apple silicon up to
the test suite; they differ from Ubuntu in section 5, because the pinned PyTorch and SciPy wheels
do not work there as shipped. The Windows steps use the same commands as Ubuntu but have not yet
been run on Windows. If a step fails, the troubleshooting table in section 13 is the first place
to look. Report anything not covered there.

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
- Brew asks `Do you want to proceed with the installation? [y/n]`. Run the install on its own and
  answer `y`; lines pasted after it are read as answers and the install is aborted.
- On a fresh Homebrew install, add Homebrew to the path of the shell you use. The prompt ends in
  `%` for zsh and `$` for bash:

```bash
echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile       # zsh
echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.bash_profile   # bash
eval "$(/opt/homebrew/bin/brew shellenv)"
```

- Use Homebrew's Python 3.10, `/opt/homebrew/bin/python3.10`. The `python3` that ships with macOS
  is 3.9, and no pinned PyTorch or `contourpy` build exists for it.

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
# Ubuntu (also Git Bash on Windows, using .venv/Scripts/activate instead)
python3.10 -m venv .venv
source .venv/bin/activate
```

```bash
# macOS
/opt/homebrew/bin/python3.10 -m venv .venv
source .venv/bin/activate
python --version     # must print Python 3.10.x; if not, delete .venv and create it again
```

The prompt should now start with `(.venv)`.

## 5. Install the pinned packages

On **Windows and Ubuntu**, install the pinned requirements directly:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e . --no-deps
```

On **macOS**, three pinned packages need different handling:

- `torch==2.14.0+cpu` cannot be found, because PyTorch's CPU index does not carry macOS builds.
  PyTorch is installed from the default index instead.
- The prebuilt SciPy 1.15.3 wheel does not load on macOS 27 (`dlopen` rejects `_spropack` with a
  `__thread_bss` section error). Newer SciPy needs Python 3.11, so SciPy 1.15.3 is built from
  source against Apple's Accelerate framework. Building it against Homebrew's OpenBLAS instead
  loads a second OpenMP runtime next to PyTorch's and aborts the process with
  `OMP: Error #15`. The build takes roughly 10 to 30 minutes.
- The requirements are a complete pinned set, so they are installed with `--no-deps`, then
  checked with `pip check`.

```bash
brew install gcc
python -m pip install --upgrade pip
pip install torch==2.14.0 torchvision==0.29.0
grep -v -E "^(torch|torchvision|scipy)==|extra-index-url" requirements.txt > requirements-macos.txt
pip install --no-cache-dir --no-deps -r requirements-macos.txt
rm requirements-macos.txt
pip install --no-binary scipy --no-cache-dir scipy==1.15.3 \
  -Csetup-args=-Dblas=accelerate -Csetup-args=-Dlapack=accelerate
pip install -e . --no-deps
pip check
python -c "import scipy.sparse.linalg, torch, pkg_resources; print('imports ok')"
```

`pip check` must print `No broken requirements found.` and the last line must print `imports ok`.

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
| `No matching distribution found for torch==2.14.0` (versions only up to 2.8.0), or `contourpy` `Requires-Python >=3.10` | The virtual environment was made with Python 3.9 | Delete `.venv` and create it with Python 3.10 (section 4) |
| `ImportError: dlopen ... _spropack ... __thread_bss` on macOS | Prebuilt SciPy 1.15.3 wheel does not load on macOS 27 | Build SciPy from source against Accelerate (section 5) |
| `OMP: Error #15: Initializing libomp.dylib, but found libomp.dylib already initialized`, then `Abort trap: 6` | SciPy built against Homebrew OpenBLAS, so two OpenMP runtimes load | Rebuild SciPy against Accelerate (section 5) |
| `ModuleNotFoundError: No module named 'pkg_resources'` | setuptools newer than the pinned 79.0.1 | `pip install setuptools==79.0.1` |
| `ResolutionImpossible` naming `numpy==2.2.6` | pip resolver failure on the full pinned set | Install with `--no-deps`, then `pip check` (section 5) |
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
