@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM ============================================================================
REM reset_data.bat
REM
REM Purpose:
REM   Reset ("wipe") generated / temporary shot-processing data so you can re-run
REM   the pipeline from a clean state.
REM
REM What it deletes (by default):
REM   - work\   (all per-run temp artifacts)
REM   - data\   (global tables / datasets)
REM   - media\exports and media\previews (exported/generated media)
REM   - legacy videos\ folder (from the old layout, if present)
REM   - legacy root shot_labels.* (from the old layout, if present)
REM   - Python bytecode caches (__pycache__)
REM   - Python bytecode caches (__pycache__)
REM
REM What it does NOT delete (by default):
REM   - Your scripts (*.py)
REM   - Global assets like assets\hoop_ellipses.json (unless you opt-in below)
REM
REM Safety:
REM   - Asks for confirmation before deleting anything
REM ============================================================================

REM Move to the directory where this .bat lives (so paths work even if launched
REM from a different folder).
pushd "%~dp0"

echo.
echo ==========================================================
echo RESET DATA - THIS WILL DELETE GENERATED PIPELINE OUTPUTS
echo ==========================================================
echo It will remove:
echo   - work\
echo   - data\
echo   - media\exports\
echo   - media\previews\
echo   - videos\   (legacy)
echo   - shot_labels.* in repo root (legacy)
echo   - __pycache__ folders
echo.

REM Optional: set to 1 if you also want to delete assets\hoop_ellipses.json (hoop ellipse data).
set "RESET_CALIBRATION=0"

REM Ask the user to confirm (CHOICE sets ERRORLEVEL: 1=Yes, 2=No).
choice /M "Proceed with deletion?"
if errorlevel 2 (
  echo.
  echo Cancelled. No files were deleted.
  popd
  exit /b 0
)

echo.
echo Deleting work\ ...
if exist "work" rmdir /s /q "work"

echo Deleting data\ ...
if exist "data" rmdir /s /q "data"

echo Deleting media exports/previews ...
if exist "media\exports" rmdir /s /q "media\exports"
if exist "media\previews" rmdir /s /q "media\previews"

echo Deleting legacy videos\ (old layout) ...
if exist "videos" rmdir /s /q "videos"

echo Deleting legacy label outputs in repo root (old layout) ...
del /f /q "shot_labels.csv" 2>nul
del /f /q "shot_labels.json" 2>nul
del /f /q "clip_labels.csv" 2>nul
del /f /q "clip_labels.json" 2>nul
del /f /q "ellips.json" 2>nul

REM Optional: delete hoop ellipse store if you really want a full reset.
if "%RESET_CALIBRATION%"=="1" (
  echo RESET_CALIBRATION=1, deleting assets\hoop_ellipses.json ...
  del /f /q "assets\hoop_ellipses.json" 2>nul
)

echo Deleting __pycache__ folders ...

REM Remove Python cache folders recursively from the repo.
for /d /r %%D in (__pycache__) do (
  if exist "%%D" rmdir /s /q "%%D"
)

echo.
echo Recreating empty folders (so pipeline has expected paths) ...

mkdir "work" 2>nul
mkdir "work\inputs" 2>nul
mkdir "work\runs" 2>nul
mkdir "data" 2>nul
mkdir "data\ball_tracking" 2>nul
mkdir "assets" 2>nul
mkdir "media" 2>nul
mkdir "media\exports" 2>nul
mkdir "media\previews" 2>nul

echo.
echo Done. Data reset complete.

popd
exit /b 0

