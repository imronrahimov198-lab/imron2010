@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
 echo Run INSTALL_WINDOWS.bat first.
 pause
 exit /b 1
)
".venv\Scripts\python.exe" -m pip install pyinstaller
if errorlevel 1 exit /b 1
set QT_API=pyside6
".venv\Scripts\python.exe" -m PyInstaller --noconfirm --clean --onedir --windowed --name StoneReliefStudioPro --collect-all vtkmodules --collect-all pyvista --collect-all pyvistaqt --collect-all scipy --collect-all shapely --hidden-import PySide6.QtSvg --hidden-import PySide6.QtOpenGLWidgets stone_relief_studio_fixed.py
if errorlevel 1 (
 echo Build failed. Read the output above.
 pause
 exit /b 1
)
echo Output: dist\StoneReliefStudioPro\StoneReliefStudioPro.exe
echo Copy the entire output folder, including DLL files.
pause
