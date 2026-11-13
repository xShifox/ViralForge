# ViralForge - install, start, and match the reference machine

<!-- BUNDLE VERSION 6 -->

Read this whole file, then follow it in order. Do not skip ahead and do not stop
part way. It is written for an AI agent with terminal access.

If this comment does not say **BUNDLE VERSION 6**, you have an older copy. Ask
for a fresh one before continuing.

---

## 0. What you are installing

ViralForge turns one long video into short vertical clips (9:16) with burned-in
subtitles and a headline, ready for TikTok, YouTube Shorts and Instagram Reels.
It runs entirely on the user's machine. No account, no server, nothing uploaded.

**It needs an NVIDIA graphics card.** Both speech recognition and video encoding
run on it. The card is detected and used as far as its memory allows, so any
recent NVIDIA card works - a 4 GB card is slower, not unsupported. Without a
card the tool stops with an explanation rather than running on the processor.

## Ground rules

- Work only inside the folder you extract this bundle into. Do not delete, move
  or modify anything else on the machine.
- Never upload, sync or send any of the user's files anywhere.
- Expect roughly **5 GB of downloads**: ~2 GB of Python packages, plus a ~3 GB
  speech model. It is quiet for long stretches. That is not a hang.
- Check what is already installed rather than reinstalling it, and confirm each
  step succeeded before starting the next.
- If a step fails, read the actual error and fix the cause. Do not skip a step
  and hope, and do not work around a failure by editing the thing being checked.
- Report a difference between this machine and the reference as a finding. Do not
  edit `data\settings.json` or the code to make a check pass - that turns a real
  report into a false one while leaving the problem in place.

---

## 1. Shell reference

You may be in Command Prompt or PowerShell. Use the column that matches your
prompt.

| Task | Command Prompt | PowerShell |
|---|---|---|
| Change directory | `cd /d C:\ViralForge` | `Set-Location C:\ViralForge` |
| Expand a variable | `%VAR%` | `$env:VAR` |
| Search recursively | `where /r "C:\dir" file.exe` | `Get-ChildItem C:\dir -Recurse -Filter file.exe` |

`dir`, `python`, `ffmpeg` and `winget` work in both.

### Define PY first

A freshly installed Python is not on the PATH of an already-open terminal, and
hard-coding a path is how you end up with two different Pythons in one session.

Command Prompt:

```
set "PY=python"
```

PowerShell:

```
$env:PY = "python"
```

Prove it before going further:

```
%PY% --version
```
```
& $env:PY --version
```

A version means `PY` is right. An error means find Python (step 3) and set `PY`
to the real path.

---

## 2. Extract this bundle

You are reading this file from inside the bundle, so if you can read it the
extraction already happened. Note which folder you are in and use it below. If
you somehow only have the archive, extract it to `C:\ViralForge` — and be aware
`Expand-Archive` reads `.zip` only, not `.rar` or `.7z`; for those use 7-Zip, or
`tar -xf`, or ask the user to extract it.

Confirm these exist, in the folder that contains this file:

```
requirements.txt
web\dist\index.html
```

```
Get-ChildItem -Recurse -Filter requirements.txt
Get-ChildItem -Recurse -Filter index.html
```

`web\dist\index.html` matters most: it is the built interface, and without it
the app starts and then shows an empty window.

---

## 3. Check what is already installed

```
where python
where ffmpeg
where winget
```

**Do not trust `where python` on its own.** On a fresh Windows machine it can
match the Microsoft Store stub, where a successful result means the Store opens
rather than that Python exists. The only real check is:

```
python --version
```

A genuine Python prints a version. If the Store opens instead, treat it as
absent. Do the same for ffmpeg with `ffmpeg -version`.

Expect Python and ffmpeg to be both missing.

### Install Python 3.13, if needed

```
winget install --id Python.Python.3.13 --exact --source winget --accept-package-agreements --accept-source-agreements --silent
```

Then set `PY` to the real path, checking both locations — a machine-wide install
is not under `LOCALAPPDATA`:

```
"%LOCALAPPDATA%\Programs\Python\Python313\python.exe"
"C:\Program Files\Python313\python.exe"
```

If neither exists:

```
where /r "C:\Program Files" python.exe
```

Set `PY` to what you find, then confirm with `%PY% --version`.

### Install ffmpeg, if needed

**This is the step people miss.** ffmpeg is a separate program, not a Python
package. `requirements.txt` only mentions it in a comment, so pip will not
provide it, and ViralForge refuses to start without it.

```
winget install --id Gyan.FFmpeg --exact --source winget --accept-package-agreements --accept-source-agreements --silent
```

The winget build puts its binaries here:

```
"%LOCALAPPDATA%\Microsoft\WinGet\Links\ffmpeg.exe" -version
```

If that path is wrong:

```
where /r "%LOCALAPPDATA%\Microsoft\WinGet" ffmpeg.exe
```

If ffmpeg cannot be made to work, stop and say so clearly — everything after
this will fail without it.

---

## 4. The Python packages

```
%PY% -m pip install --upgrade pip
%PY% -m pip install -r requirements.txt
```

```
& $env:PY -m pip install --upgrade pip
& $env:PY -m pip install -r requirements.txt
```

The long part, ~2 GB. Most of that is the NVIDIA CUDA libraries, and pip pulls
them in on Windows whether or not the machine has an NVIDIA card. They are
required, not optional - do not uninstall them later to "clean up".

If pip fails on permissions, say you need to run it once as Administrator, or
retry with `--user`.

---

## 5. Verify

```
%PY% --version
ffmpeg -version
%PY% -c "import fastapi, uvicorn, faster_whisper, PIL, numpy; print('packages ok')"
%PY% -m pytest -q
```

**What matters: 0 failed.** A couple of dozen tests skip when optional sample
media is absent, and this bundle does not include it — skipping is the correct
result. The number that passes changes between releases, so there is no figure
to compare against. Report only the failure count.

---

## 6. Pre-fetch the speech model

The one genuine gap. Normally the ~3 GB model is downloaded in the middle of the
first job, where it looks like the app has hung. Do it deliberately instead:

```
%PY% scripts/match-my-setup.py --fetch
```

```
& $env:PY scripts/match-my-setup.py --fetch
```

---

## 7. Start it

**Use `--open`.** Do not run `python -m clipforge` on its own: with no flag it
opens no interface at all and holds the terminal open until interrupted.

```
%PY% -m clipforge --open
```

```
& $env:PY -m clipforge --open
```

`--open` shows it in the user's normal browser, which is better while setting up
because errors are readable there.

The flag is `--open`, **not** `--browser`. `--browser` is an alias that only
exists inside `start.bat`; the Python entry point has never had it.

The user will start it themselves in future by double-clicking **`ViralForge.vbs`**
— not `start.bat`, which is a batch file and always flashes a black console
window. `ViralForge.vbs` starts the same thing with no console.

---

## 8. Check it against the reference machine

With ViralForge still running, in a second terminal:

```
%PY% scripts/match-my-setup.py
```

```
& $env:PY scripts/match-my-setup.py
```

It prints what this machine reports, checks the settings that must match, and
lists any differences that are expected. It should end in **MATCHED**.

A **WARN** about a remembered output folder that does not exist here is not a
failure — that field is machine-specific by definition and ships blank. The user
sets their own when they first drop a video in.

**If a check reports a difference in a setting, that is a real problem: report
it rather than editing the file being checked.** Editing it to silence a check
makes the report a lie while the problem stays.

---

## 9. Prove it end to end

Process one short disposable video — a minute or two, not the user's real work.
Confirm:

- the job reaches **completed**
- the clips are written to the folder the user chose at the prompt, **not** into
  this bundle's `data\jobs`
- a clip shows burned-in subtitles
- the pill along the top is green **ffmpeg**, and the speech pill reads
  **`GPU - <mode>`** with a `fast`, `balanced` or `light` badge

If the clips land in `data\jobs` instead of the chosen folder, the output folder
did not take. That is a genuine bug - report it.

The speech pill is the one to watch. It must say `GPU`, never `CPU` and never
`blocked`: this build does not run speech recognition on the processor, and a job
that cannot use the graphics card stops with a reason instead. A `light` badge
with `GPU - int8` on a 4 GB card is correct and expected - see the next section.

---

## What "the same as the reference machine" means

**An NVIDIA graphics card is required.** Not preferred - required. This build
runs speech recognition and video encoding on the graphics card, and when it
cannot, it stops and says why rather than quietly using the CPU. That is
deliberate: the tool is several times faster on the card, and a silent fallback
is how a machine ends up doing the wrong thing for weeks while looking fine.

So if step 8 reports `speech_on_gpu` or `encoder_on_gpu` as a difference, that is
a real fault. Do not switch the settings to make the check pass. Report it.

**Different cards are equally correct.** The reference machine has an NVIDIA RTX
4060 Laptop with 8 GB. A smaller card is not a worse install - it works out how
much the card can hold and does that much.

The card's name is checked first, because a card's model number is more reliable
than the memory a driver chooses to report. A GTX 1650, 1660 or 16xx, or a GT
1030/1050, is treated as a 4 GB card whatever it reports:

| Card | Mode | Speech runs in |
|---|---|---|
| A GTX 1650 / 1660 / 16xx, or a GT 1030 / 1050 | `light` | `int8` |
| Any other card with 8 GB or more | `fast` | `float16` |
| Any other card with 6 GB to 8 GB | `balanced` | `int8_float16` |
| Any other card under 5 GB | `light` | `int8` |

A 4 GB card such as a GTX 1650 cannot hold the speech model in `float16` at all,
so it runs `int8` and takes longer. **The exported video is the same either
way.** The model, the 1080x1920 output, the subtitles, the encoder's quality
target, profile and level are identical on every machine; only the speed differs.
Expect a `GPU - int8` chip with a `light` badge rather than the reference's
`GPU - float16` with `fast`, and treat that as correct.

That last claim was measured rather than assumed. Both ends of the range were
encoded from the same clip through the same filter chain and scored against a
lossless master with VMAF: **99.95** for the 4 GB settings, **99.94** for the
8 GB settings, a difference of 0.01 points and 1.1% in file size. Both are
indistinguishable from the original. Re-run it yourself with
`python scripts\audit_tier_quality.py` - it exits non-zero if the two ever drift
apart by more than two points.

**Do not uninstall the NVIDIA CUDA packages.** They are required. pip installs
them on Windows whether or not a card is present, which is why step 4 downloads
so much.

These are settings rather than hardware, and they already match - the
`data/settings.json` here is a copy of the reference machine's, with the output
folder blanked because a path from the builder's disk belongs on nobody else's
machine. **Do not edit it.**

| Setting | Value |
|---|---|
| Speech model | `large-v3` |
| Speech device | graphics card only (`require_gpu`) |
| Encoding quality (crf) | `19` |
| Encoder mode | `auto` - graphics card only (`require_gpu_encoder`) |
| Platforms | TikTok, YouTube Shorts, Instagram Reels |
| Finished clips | asked for per job, then moved out of the app folder |

---

## Notes

**The chip along the top reports speech recognition, not video encoding** - they
are separate subsystems, and both are separately required. On a healthy machine
it reads `GPU - <mode>` with a `fast`, `balanced` or `light` badge. The badge is
the card's mode, not a quality rating: `light` means less memory, not a worse
result.

**If the chip reads `blocked`,** speech recognition cannot use the card and will
not fall back to the CPU. Hover it for the reason. Usually it is not enough free
video memory - close other apps that use the graphics card and start again.

**A slow first real job is expected** if you skipped step 6 - that is the model
downloading.

**Where data goes:** `data\uploads`, `data\jobs`, `data\models`, `data\cache`.
Nothing leaves the machine. A prompt asks where to save finished clips before
each job, and they are moved there when it finishes - clips will not appear in
`data\jobs`.

**Finished clips move off the drive ViralForge is on, if you choose a folder on a
different one.** The move is a copy in that case, so a large job takes a little
longer to finish and needs room in both places for a moment. The clips, their
subtitles and their poster images all move together, and the download buttons
keep working afterwards.

**The speech model is a different precision on a smaller card, and that is the
one place the words can differ.** `float16` and `int8` are different arithmetic,
so a transcript can come out with a word changed here and there on a 4 GB card,
and the clip boundaries and captions follow from the transcript. The structure,
the length, the quality and every marketplace rule are the same; the occasional
individual word is not guaranteed identical. If that matters for a specific
video, run it on a machine with 8 GB.

**For more detail:** `%PY% -m clipforge --open --log-level debug`

**Do not run anything in `scripts\`** - it only holds the icon generators.
