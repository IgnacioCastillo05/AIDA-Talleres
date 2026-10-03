@echo off
setlocal
cd /d "%~dp0"
echo ============================================================
echo  AI Generative Workshops - Computer Check
echo ============================================================
if exist .venv\Scripts\python.exe (
  .venv\Scripts\python.exe check_computer.py
) else (
  py check_computer.py
)
echo.
echo Result saved as hardware_profile.json
pause
