@echo off
setlocal
cd /d "%~dp0"
where py >nul 2>nul
if errorlevel 1 (
 echo Install 64-bit Python 3.12 from python.org, including the Python Launcher.
 pause
 exit /b 1
)
if not exist ".venv\Scripts\python.exe" py -3.12 -m venv .venv
if not exist ".venv\Scripts\python.exe" (
 echo Failed to create the Python environment.
 pause
 exit /b 1
)
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (
 echo Installation failed. Check the error above and your Internet connection.
 pause
 exit /b 1
)
echo Installation complete. Run START_WINDOWS.bat.
pause
