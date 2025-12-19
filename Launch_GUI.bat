@echo off
REM ============================================
REM WarzaVision Pro 4.0 - Manual GUI Launcher
REM Use if GUI doesn't auto-launch from GTuner
REM ============================================

title WarzaVision GUI Launcher
color 0A

echo.
echo  ============================================
echo   WarzaVision Pro 4.0 - GUI Launcher
echo  ============================================
echo.

REM Check for Python
where python >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python not found!
    echo Run INSTALL.bat first.
    pause
    exit /b 1
)

REM Check for PyQt6
python -c "import PyQt6" 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [INFO] PyQt6 not installed. Installing now...
    pip install PyQt6
)

REM Try to read port from file
set PORT=59420
set PORT_FILE=%~dp0.warzavision_port
if exist "%PORT_FILE%" (
    set /p PORT=<"%PORT_FILE%"
    echo [INFO] Found saved port: %PORT%
)

echo.
echo  Current Settings:
echo  -----------------
echo  Port: %PORT%
echo.
echo  Options:
echo  1. Use saved port (%PORT%)
echo  2. Enter custom port
echo  3. Use default port (59420)
echo.

set /p CHOICE="Enter choice (1-3): "

if "%CHOICE%"=="2" (
    set /p PORT="Enter port number: "
)
if "%CHOICE%"=="3" (
    set PORT=59420
)

echo.
echo [INFO] Starting GUI on port %PORT%...
echo.
echo  NOTE: Make sure GTuner is running with Warzatools2K.py loaded.
echo  The GUI will auto-connect when the CV script is active.
echo.

REM Launch GUI
python "%~dp0WarzaGUI_Qt.py" %PORT%

pause
