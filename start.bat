@echo off
title AuraAssess Startup Launcher
echo ==================================================
echo             AURAASSESS STARTUP LAUNCHER          
echo ==================================================
echo.
echo [1/2] Opening Web Dashboard in browser...
start http://localhost:8080

echo [2/2] Launching Python FastAPI Web Server...
echo (Press Ctrl+C in this window to stop the server)
echo.
python src/server.py

pause
