@echo off
setlocal EnableExtensions EnableDelayedExpansion

if "%~1"=="" (
  set "NAME_RAW=test1"
) else (
  set "NAME_RAW=%~1"
)
REM Sanitize NAME so users can pass "test1", "test1.mp4", or even "videos\test1"
for %%F in ("%NAME_RAW%") do set "NAME=%%~nF"
echo NAME_RAW=%NAME_RAW%
echo NAME=%NAME%

REM Always run from the repo root (the directory containing this .bat)
pushd "%~dp0.."

REM 0) Ensure required directories exist (new folder layout)
if not exist "data" mkdir "data"
if not exist "data\ball_tracking" mkdir "data\ball_tracking"
if not exist "assets" mkdir "assets"
if not exist "assets\hoop_ellipses" mkdir "assets\hoop_ellipses"
if not exist "media" mkdir "media"
if not exist "media\exports" mkdir "media\exports"
if not exist "media\previews" mkdir "media\previews"
if not exist "work" mkdir "work"
if not exist "work\inputs" mkdir "work\inputs"
if not exist "work\runs" mkdir "work\runs"
if not exist "work\frames_raw" mkdir "work\frames_raw"
if not exist "work\frames_batch" mkdir "work\frames_batch"
if not exist "work\frames_hsv" mkdir "work\frames_hsv"
if not exist "work\debug" mkdir "work\debug"
if not exist "work\debug\labels" mkdir "work\debug\labels"
if not exist "work\debug\goal_detector" mkdir "work\debug\goal_detector"
if not exist "work\segmentation" mkdir "work\segmentation"

REM 1) Download (remote file is always videos/test1.mp4)
scp weatherwolf@192.168.2.29:Documents/Projects/basketballtracker/videos/test1.mp4 "work\inputs\%NAME%.mp4"
if errorlevel 1 (
  echo SCP failed. Stopping.
  popd
  exit /b 1
)

REM 2) Confirm file exists locally
if not exist "work\inputs\%NAME%.mp4" (
  echo File "%NAME%.mp4" not found after SCP. Stopping.
  popd
  exit /b 1
)

REM 2.5) Choose a batch_id:
REM   - If MP4 has creation_time metadata, use it (stable across reruns of same source)
REM   - Otherwise, use a fingerprint derived from (num_clips, frames_per_clip in order) after filtering
set "MP4_CREATION_TIME="
for /f "usebackq delims=" %%A in (`ffprobe -v error -show_entries format_tags^=creation_time -of default^=nw^=1:nk^=1 "work\\inputs\\%NAME%.mp4" 2^>nul`) do (
  set "MP4_CREATION_TIME=%%A"
)

set "BATCH_SRC=fingerprint"
set "BATCH_ID=pending_%RANDOM%%RANDOM%"
if not "!MP4_CREATION_TIME!"=="" (
  set "BATCH_SRC=mp4_creation_time"
  set "BATCH_ID=!MP4_CREATION_TIME!"
  REM sanitize to a Windows-safe-ish folder name
  set "BATCH_ID=!BATCH_ID: =_!"
  set "BATCH_ID=!BATCH_ID::=-!"
  set "BATCH_ID=!BATCH_ID:T=_!"
  set "BATCH_ID=!BATCH_ID:Z=!"
  set "BATCH_ID=!BATCH_ID:.=!"
)

set "RUN_DIR=work\runs\!BATCH_ID!"
REM If we're re-running the same batch_id, wipe the run dir so it doesn't accumulate junk.
if exist "!RUN_DIR!" rmdir /s /q "!RUN_DIR!"
if not exist "!RUN_DIR!" mkdir "!RUN_DIR!"
if not exist "!RUN_DIR!\inputs" mkdir "!RUN_DIR!\inputs"
if not exist "!RUN_DIR!\frames_raw" mkdir "!RUN_DIR!\frames_raw"
if not exist "!RUN_DIR!\frames_batch" mkdir "!RUN_DIR!\frames_batch"
if not exist "!RUN_DIR!\debug" mkdir "!RUN_DIR!\debug"
if not exist "!RUN_DIR!\debug\labels" mkdir "!RUN_DIR!\debug\labels"
if not exist "!RUN_DIR!\debug\goal_detector" mkdir "!RUN_DIR!\debug\goal_detector"

move /Y "work\inputs\%NAME%.mp4" "!RUN_DIR!\inputs\%NAME%.mp4" >nul

REM 3) Extract frames (note: %%06d is required in .bat)
ffmpeg -y -i "!RUN_DIR!\inputs\%NAME%.mp4" "!RUN_DIR!\frames_raw\frame_%NAME%_%%06d.jpg"
if errorlevel 1 (
  echo FFMPEG failed. Stopping.
  REM Cleanup empty run folder on failure
  if exist "!RUN_DIR!" rmdir /s /q "!RUN_DIR!"
  popd
  exit /b 1
)

REM 4) Filter and detect
python dev\filter_frames.py --frames-raw-dir "!RUN_DIR!\frames_raw" --frames-batch-dir "!RUN_DIR!\frames_batch"
if errorlevel 1 exit /b 1

REM 4.5) If no MP4 timestamp, compute fingerprint batch_id from shots_manifest.json and rename run folder
if /I "!BATCH_SRC!"=="fingerprint" (
  set "MANIFEST=!RUN_DIR!\frames_batch\shots_manifest.json"
  if exist "!MANIFEST!" (
    set "FINAL_ID="
    for /f "usebackq delims=" %%A in (`python -c "import json,sys,hashlib; p=sys.argv[1]; m=json.load(open(p,'r',encoding='utf-8')); ds=m.get('datasets',[]); ds=sorted(ds,key=lambda d:int(d.get('shot_index',0))); lens=[int(d.get('num_frames',0)) for d in ds]; s='c=%d|lens=%s' % (len(lens), ','.join(str(x) for x in lens)); h=hashlib.sha1(s.encode('utf-8')).hexdigest()[:12]; print('c%d_%s' % (len(lens), h))" "!MANIFEST!"`) do (
      set "FINAL_ID=%%A"
    )
    if not "!FINAL_ID!"=="" (
      REM Overwrite-by-batch-id: if the target exists, wipe it then move this run into place.
      if exist "work\runs\!FINAL_ID!" rmdir /s /q "work\runs\!FINAL_ID!"
      move "!RUN_DIR!" "work\runs\!FINAL_ID!" >nul
      set "BATCH_ID=!FINAL_ID!"
      set "RUN_DIR=work\runs\!BATCH_ID!"
    )
  )
)

echo.
echo Batch id: !BATCH_ID!  (source=!BATCH_SRC!)
echo Run dir:  !RUN_DIR!
echo.

REM If filtering produced no shot folders, delete the empty run folder and stop.
set "HAS_SHOTS=0"
for /d %%D in ("!RUN_DIR!\frames_batch\*") do (
  set "HAS_SHOTS=1"
  goto :has_shots_done
)
:has_shots_done
if "!HAS_SHOTS!"=="0" (
  echo No shot folders produced for this run. Cleaning up run folder.
  if exist "!RUN_DIR!" rmdir /s /q "!RUN_DIR!"
  popd
  exit /b 2
)

REM 5) Label new shot folders (interactive). Pass "nolabel" to skip.
if /I "%~2" NEQ "nolabel" (
  REM Policy A: exports are stored under media/exports/<batch_id> and overwritten on rerun.
  set "EXPORT_DIR=media\exports\!BATCH_ID!"
  if exist "!EXPORT_DIR!" rmdir /s /q "!EXPORT_DIR!"
  mkdir "!EXPORT_DIR!" 2>nul

  REM Ellipse storage consistent with exports: assets/hoop_ellipses/<batch_id>/...
  set "ELLIPSE_BATCH_DIR=assets\hoop_ellipses\!BATCH_ID!"
  if exist "!ELLIPSE_BATCH_DIR!" rmdir /s /q "!ELLIPSE_BATCH_DIR!"
  mkdir "!ELLIPSE_BATCH_DIR!" 2>nul
  mkdir "!ELLIPSE_BATCH_DIR!\per_shot" 2>nul
  REM Seed the per-batch global ellipse file from the repo-level default if it exists.
  if exist "assets\hoop_ellipses.json" (
    copy /Y "assets\hoop_ellipses.json" "!ELLIPSE_BATCH_DIR!\global.json" >nul
  )

  python dev\label_shots.py ^
    --frames-dir "!RUN_DIR!\frames_batch" ^
    --videos-dir "!EXPORT_DIR!" ^
    --debug-dir "!RUN_DIR!\debug\labels" ^
    --ellipse-meta-dir "!ELLIPSE_BATCH_DIR!\per_shot" ^
    --global-ellipse "!ELLIPSE_BATCH_DIR!\global.json" ^
    --out-base data/shot_labels ^
    --open --export-debug-frames --relabel --overwrite-preview --overwrite-debug-frames
  if errorlevel 1 (
    popd
    exit /b 1
  )
)

REM 6) Extract ball tracking data for all labeled shots
python dev\extract_ball_tracking.py
if errorlevel 1 (
  popd
  exit /b 1
)

REM 7) Verify (or refit) the hoop ellipse for this batch
python dev\fit_ellipse.py --batch !BATCH_ID!
if errorlevel 1 (
  popd
  exit /b 1
)

popd
endlocal
