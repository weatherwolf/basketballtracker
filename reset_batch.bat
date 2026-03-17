@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM ============================================================================
REM reset_batch.bat
REM
REM Purpose:
REM   Prepare for a NEW BATCH while keeping your GLOBAL archive.
REM
REM Deletes (working dirs only):
REM   - work\runs\*   (all per-run temp artifacts: inputs, frames, debug)
REM
REM Keeps (global storage):
REM   - data\*            (global labels, ball_tracking, etc.)
REM   - assets\*          (global hoop ellipse store, etc.)
REM   - media\exports\*   (exported goal/miss mp4s)
REM ============================================================================

pushd "%~dp0"

echo.
echo ==========================================================
echo RESET BATCH - clears working dirs only (keeps global archive)
echo ==========================================================
echo It will remove:
echo   - work\runs\*
echo.

choice /M "Proceed?"
if errorlevel 2 (
  echo.
  echo Cancelled. No files were deleted.
  popd
  exit /b 0
)

echo.
echo Deleting working folders ...
if exist "work\runs" rmdir /s /q "work\runs"

echo Recreating empty folders ...
mkdir "work" 2>nul
mkdir "work\runs" 2>nul

echo.
echo Done. You can now run pipeline.bat for the next batch.
echo.

popd
exit /b 0

