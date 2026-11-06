@echo off
title AuraAssess - Multimodal Behavioral AI (Port 8080)
cd /d "%~dp0"
echo ============================================================
echo      AURAASSESS - MULTIMODAL INTERVIEW AI (PORT 8080)
echo ============================================================
echo.
echo Local Live URL: http://localhost:8080
echo.
python src/server.py
pause
