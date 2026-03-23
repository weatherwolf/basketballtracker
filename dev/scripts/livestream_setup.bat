@echo off
setlocal

pushd "%~dp0..\.."

set "PI_USER=weatherwolf"
set "PI_HOST=192.168.2.29"
set "PI_DIR=Documents/Projects/basketballtracker"
set "LOCAL_MP4=work\inputs\camera_check.mp4"

if not exist "work\inputs" mkdir "work\inputs"

REM 1) Record a camera check clip on the Pi (same settings as inference.py)
echo Recording 1 second camera check clip on Pi...
ssh %PI_USER%@%PI_HOST% "cd %PI_DIR% && python record_camera_check.py --duration 1"
if errorlevel 1 (
  echo SSH failed. Stopping.
  popd
  exit /b 1
)

REM 2) Download the clip
echo Downloading camera_check.mp4 from Pi...
scp %PI_USER%@%PI_HOST%:%PI_DIR%/videos/manual_camera_check.mp4 "%LOCAL_MP4%"
if errorlevel 1 (
  echo SCP failed. Stopping.
  popd
  exit /b 1
)

REM 3) Fit the ellipse from sticker detection
echo Fitting ellipse from stickers...
python dev\utils\fit_ellipse.py --sticker-check-silent
if errorlevel 1 (
  echo Ellipse fitting failed or cancelled. Stopping.
  popd
  exit /b 1
)

REM 4) Upload ellipse.json to the Pi
echo Uploading ellipse.json to Pi...
scp assets\hoop_ellipses.json %PI_USER%@%PI_HOST%:%PI_DIR%/ellipse.json
if errorlevel 1 (
  echo SCP failed. Stopping.
  popd
  exit /b 1
)

echo.
echo Setup complete. The Pi is ready for inference.
echo Run on Pi: python inference.py

popd
endlocal
