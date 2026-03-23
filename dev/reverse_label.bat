@echo off
pushd "%~dp0.."
python dev\reverse_label.py %*
popd
