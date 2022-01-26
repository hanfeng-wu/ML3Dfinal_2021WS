@echo off
pushd %~dp0

blender -b -P generator.py

popd
