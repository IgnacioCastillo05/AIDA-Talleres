@echo off
setlocal
cd /d "%~dp0"
echo ============================================================
echo  AI Generative Workshops - Environment Setup
echo ============================================================
where py >nul 2>nul
if errorlevel 1 (
  echo Python launcher ^(py^) was not found.
  echo Install Python 3.10 or 3.11 and run this file again.
  pause
  exit /b 1
)
if not exist .venv\Scripts\python.exe (
  echo Creating .venv ...
  py -m venv .venv
)
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
if errorlevel 1 (
  echo.
  echo Installation did not finish successfully.
  echo Read the error above. If PyTorch installation fails, use the official
  echo PyTorch selector for your computer and then re-run Check_Computer.bat.
  pause
  exit /b 1
)
echo.
echo Setup finished. Running hardware check now ...
python check_computer.py
pause
