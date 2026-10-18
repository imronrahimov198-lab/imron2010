@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
 echo Run INSTALL_WINDOWS.bat first.
 pause
 exit /b 1
)
".venv\Scripts\python.exe" stone_relief_studio_fixed.py %*
if errorlevel 1 pause
