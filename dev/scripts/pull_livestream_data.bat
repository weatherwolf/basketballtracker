@echo off
setlocal EnableExtensions EnableDelayedExpansion

set "PI_HOST=weatherwolf@192.168.2.29"
set "PI_REPO=Documents/Projects/basketballtracker"

REM Always run from the repo root
pushd "%~dp0..\.."

echo Running the pull script

REM -----------------------------------------------------------------------
REM 1) Find the live batch ID on the Pi
REM -----------------------------------------------------------------------
set "BATCH_ID="
echo Set batch id
for /f "usebackq delims=" %%A in (`ssh %PI_HOST% "ls %PI_REPO%/work/runs/ 2>/dev/null"`) do (
    echo %%A | findstr /R "^live_" >nul 2>&1
    if not errorlevel 1 set "BATCH_ID=%%A"
)

echo After the forloop

if "!BATCH_ID!"=="" (
    echo No live batch found on Pi.
    popd
    exit /b 1
)

echo Found batch: !BATCH_ID!
echo.

REM -----------------------------------------------------------------------
REM 2) Set up local directories
REM -----------------------------------------------------------------------
set "RUN_DIR=work\runs\!BATCH_ID!"
if exist "!RUN_DIR!" rmdir /s /q "!RUN_DIR!"
mkdir "!RUN_DIR!\frames_batch" 2>nul
if not exist "data" mkdir "data"

REM -----------------------------------------------------------------------
REM 3) Pull labels first — needed to know which shot folders to fetch
REM -----------------------------------------------------------------------
echo Pulling labels...
scp %PI_HOST%:%PI_REPO%/data/shot_labels.json "data\shot_labels_live_tmp.json"
if errorlevel 1 (
    echo SCP labels failed.
    popd & exit /b 1
)

REM -----------------------------------------------------------------------
REM 4) Pull only labeled (stage-2) shot folders
REM -----------------------------------------------------------------------
echo Pulling labeled shots...
set "SHOT_COUNT=0"
for /f "usebackq delims=" %%A in (`python dev\utils\list_live_shots.py !BATCH_ID!`) do (
    scp -r %PI_HOST%:%PI_REPO%/work/runs/!BATCH_ID!/frames_batch/%%A "!RUN_DIR!\frames_batch"
    if errorlevel 1 (
        echo SCP shot %%A failed.
        popd & exit /b 1
    )
    set /a SHOT_COUNT+=1
)
echo Pulled !SHOT_COUNT! labeled shots.

REM -----------------------------------------------------------------------
REM 5) Pull ellipse.json -> assets/hoop_ellipses/<batch_id>/global.json
REM -----------------------------------------------------------------------
echo Pulling ellipse...
set "ELLIPSE_BATCH_DIR=assets\hoop_ellipses\!BATCH_ID!"
if exist "!ELLIPSE_BATCH_DIR!" rmdir /s /q "!ELLIPSE_BATCH_DIR!"
mkdir "!ELLIPSE_BATCH_DIR!\per_shot" 2>nul

scp %PI_HOST%:%PI_REPO%/ellipse.json "!ELLIPSE_BATCH_DIR!\global.json"
if errorlevel 1 (
    echo SCP ellipse failed.
    popd & exit /b 1
)

REM -----------------------------------------------------------------------
REM 6) Create per-shot ellipse files and merge labels into local shot_labels
REM -----------------------------------------------------------------------
echo Merging labels and creating per-shot ellipse files...
python dev\utils\pull_merge.py !BATCH_ID!
if errorlevel 1 (
    echo Label merge failed.
    popd & exit /b 1
)

REM -----------------------------------------------------------------------
REM 7) Pull exported mp4s (non-fatal — may be empty if no goal/miss shots)
REM -----------------------------------------------------------------------
echo Pulling exports...
set "EXPORT_DIR=media\exports\!BATCH_ID!"
if exist "!EXPORT_DIR!" rmdir /s /q "!EXPORT_DIR!"
mkdir "!EXPORT_DIR!" 2>nul
scp -r %PI_HOST%:%PI_REPO%/media/exports/!BATCH_ID! "media\exports" >nul 2>&1

REM -----------------------------------------------------------------------
REM 8) Extract ball tracking for the live batch
REM -----------------------------------------------------------------------
echo Extracting ball tracking...
python dev\extract_ball_tracking.py --batch !BATCH_ID! --overwrite
if errorlevel 1 (
    echo extract_ball_tracking.py failed.
    popd & exit /b 1
)

echo.
echo Done. Batch !BATCH_ID! pulled and processed.
popd
endlocal
