@echo off
REM Quick end-to-end test (small subset, 1-2 epochs). Results go to outputs\quick\.
cd /d "%~dp0"
call "%~dp0train.bat" --quick
