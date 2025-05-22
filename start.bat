@echo off
:: Obtén la ruta donde está este archivo
set SCRIPT_DIR=%~dp0
cd /d "%SCRIPT_DIR%"

:: (Opcional) Activa entorno virtual si existe
if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
)

:: Ejecuta el script Python
python3.11 vitalParserLearning_GUI.py

pause
