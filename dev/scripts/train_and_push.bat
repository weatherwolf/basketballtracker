@echo off
setlocal

call "%~dp0config.bat"

pushd "%~dp0..\.."

echo Training model...
python dev\minirocket_test.py --save-model %*
if errorlevel 1 (
    echo Training failed.
    popd & exit /b 1
)

echo Pushing model to Pi...
scp raspberry\minirocket_model.joblib %PI_USER%@%PI_HOST%:%PI_DIR%/minirocket_model.joblib
if errorlevel 1 (
    echo SCP failed.
    popd & exit /b 1
)

echo Pushing competition.py to Pi...
scp raspberry\competition.py %PI_USER%@%PI_HOST%:%PI_DIR%/competition.py
if errorlevel 1 (
    echo SCP failed.
    popd & exit /b 1
)

echo Done. Model trained and pushed to Pi.
popd
endlocal
