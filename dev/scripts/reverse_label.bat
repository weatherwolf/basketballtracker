@echo off
pushd "%~dp0..\.."
python dev\utils\reverse_label.py %*
popd
