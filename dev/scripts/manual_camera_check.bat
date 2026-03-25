@echo off
setlocal

pushd "%~dp0..\.."

call "%~dp0config.bat"
set "LOCAL=work\inputs\camera_check.mp4"

if not exist "work\inputs" mkdir "work\inputs"

echo Recording camera check clip on Pi (Ctrl+C on Pi to stop early)...
ssh %PI_USER%@%PI_HOST% "cd %PI_DIR% && python record_camera_check.py"
if errorlevel 1 (
  echo SSH failed.
  popd
  exit /b 1
)

echo Downloading manual_camera_check.mp4 from Pi...
scp %PI_USER%@%PI_HOST%:%PI_DIR%/videos/manual_camera_check.mp4 "%LOCAL%"
if errorlevel 1 (
  echo SCP failed.
  popd
  exit /b 1
)

echo Opening...
start "" "%LOCAL%"

popd
endlocal
