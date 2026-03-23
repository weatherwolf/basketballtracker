@echo off
setlocal

set "PI_HOST=weatherwolf@192.168.2.29"
set "PI_REPO=Documents/Projects/basketballtracker"

pushd "%~dp0.."

echo Training model...
python dev\minirocket_test.py --save-model
if errorlevel 1 (
    echo Training failed.
    popd & exit /b 1
)

echo Pushing model to Pi...
scp raspberry\minirocket_model.joblib %PI_HOST%:%PI_REPO%/minirocket_model.joblib
if errorlevel 1 (
    echo SCP failed.
    popd & exit /b 1
)

echo Done. Model trained and pushed to Pi.
popd
endlocal
