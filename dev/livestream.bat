@echo off
setlocal EnableExtensions EnableDelayedExpansion

set "PI_USER=weatherwolf"
set "PI_HOST=192.168.2.29"
set "PI_DIR=Documents/Projects/basketballtracker"

pushd "%~dp0.."

REM 1) Setup: record calibration clip, fit ellipse, upload to Pi
echo.
echo ============================================================
echo  STEP 1 - Livestream setup (ellipse calibration)
echo ============================================================
call dev\livestream_setup.bat
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
ssh -t %PI_USER%@%PI_HOST% "cd %PI_DIR% && source venv/bin/activate && python inference.py"

REM 3) Pull data back to this machine
echo.
echo ============================================================
echo  STEP 3 - Pulling livestream data
echo ============================================================
call dev\pull_livestream_data.bat
if errorlevel 1 (
  echo Pull failed.
  popd
  exit /b 1
)

echo.
echo Done.
popd
endlocal
