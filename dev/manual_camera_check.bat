@echo off
setlocal

pushd "%~dp0.."

set "LOCAL=work\inputs\camera_check.mp4"

if not exist "work\inputs" mkdir "work\inputs"

echo Recording camera check clip on Pi (Ctrl+C on Pi to stop early)...
ssh weatherwolf@192.168.2.29 "cd Documents/Projects/basketballtracker && python record_camera_check.py"
if errorlevel 1 (
  echo SSH failed.
  popd
  exit /b 1
)

echo Downloading manual_camera_check.mp4 from Pi...
scp weatherwolf@192.168.2.29:Documents/Projects/basketballtracker/videos/manual_camera_check.mp4 "%LOCAL%"
if errorlevel 1 (
  echo SCP failed.
  popd
  exit /b 1
)

echo Opening...
start "" "%LOCAL%"

popd
endlocal
