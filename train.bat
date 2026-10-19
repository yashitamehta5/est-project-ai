@echo off
REM Full training of both models. Double-click to run. Extra arguments are passed to train.py.
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Creating the virtual environment .venv ...
    py -3.11 -m venv .venv || python -m venv .venv
    if errorlevel 1 goto :error
)

echo Installing PyTorch with CUDA 12.8 and the other requirements ...
".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\python.exe" -m pip install torch==2.11.0+cu128 torchvision==0.26.0+cu128 --index-url https://download.pytorch.org/whl/cu128
if errorlevel 1 goto :error
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto :error

echo.
".venv\Scripts\python.exe" check_gpu.py
echo.
".venv\Scripts\python.exe" train.py %*
if errorlevel 1 goto :error
echo.
echo Done. Results are in the outputs folder (see outputs\training_log.txt).
pause
exit /b 0

:error
echo.
echo Something went wrong - see the messages above.
pause
exit /b 1
