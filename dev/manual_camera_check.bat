@echo off
setlocal

pushd "%~dp0.."

set "LOCAL=work\inputs\camera_check.mp4"

if not exist "work\inputs" mkdir "work\inputs"

echo Running manual_camera_check.sh on Pi...
ssh weatherwolf@192.168.2.29 "cd Documents/Projects/basketballtracker && sh manual_camera_check.sh"
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
