@echo off
setlocal EnableExtensions EnableDelayedExpansion

call "%~dp0config.bat"
set "PUSH_CELEBRATIONS=false"
set "WHOLE_VIDEO=false"

for %%a in (%*) do (
    if /i "%%a"=="--celebrations" set "PUSH_CELEBRATIONS=true"
    if /i "%%a"=="--whole-video"  set "WHOLE_VIDEO=true"
)

pushd "%~dp0..\.."

REM 0) Optionally push celebrations to Pi
if /i "%PUSH_CELEBRATIONS%"=="true" (
    echo.
    echo ============================================================
    echo  STEP 0 - Pushing celebrations to Pi
    echo ============================================================
    for /d %%d in (raspberry\celebrations\*) do (
        ssh %PI_USER%@%PI_HOST% "mkdir -p %PI_DIR%/celebrations/%%~nd"
        scp "raspberry\celebrations\%%~nd\*.pkl" %PI_USER%@%PI_HOST%:%PI_DIR%/celebrations/%%~nd/
    )
)

REM 1) Setup: record calibration clip, fit ellipse, upload to Pi
echo.
echo ============================================================
echo  STEP 1 - Livestream setup (ellipse calibration)
echo ============================================================
call dev\scripts\livestream_setup.bat
if errorlevel 1 (
  echo Setup failed. Stopping.
  popd
  exit /b 1
)

REM 2) Run inference on the Pi interactively (-t allocates a pseudo-TTY
REM    so ANSI colours, \r status updates and the keypress listener work)
echo.
echo ============================================================
echo  STEP 2 - Starting inference on Pi  (q to quit)
echo ============================================================
set "CELEBRATIONS_FLAG="
set "WHOLE_VIDEO_FLAG="
if /i "%PUSH_CELEBRATIONS%"=="true" set "CELEBRATIONS_FLAG=--celebrations"
if /i "%WHOLE_VIDEO%"=="true"       set "WHOLE_VIDEO_FLAG=--whole-video"
ssh -t %PI_USER%@%PI_HOST% "cd %PI_DIR% && source venv/bin/activate && python inference.py --competition %CELEBRATIONS_FLAG% %WHOLE_VIDEO_FLAG%"

REM 3) Pull data back to this machine
echo.
echo ============================================================
echo  STEP 3 - Pulling livestream data
echo ============================================================
call dev\scripts\pull_livestream_data.bat
if errorlevel 1 (
  echo Pull failed.
  popd
  exit /b 1
)

echo.
echo Done.
popd
endlocal
