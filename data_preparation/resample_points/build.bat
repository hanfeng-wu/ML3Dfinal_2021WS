@echo off
pushd %~dp0

echo.
echo compiling ...
clang++ -fopenmp -ffast-math -O3 main.cpp -g -mavx -o "main.exe" || exit /b 1

echo.
echo running ...
main.exe

popd
