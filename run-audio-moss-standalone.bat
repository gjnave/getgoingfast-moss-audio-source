@echo off
setlocal EnableExtensions
cd /d "%~dp0"

set "ROOT_DIR=%~dp0"
set "VENV_DIR=%ROOT_DIR%.venv"
set "WEIGHTS_DIR=%ROOT_DIR%weights\MOSS-Audio-4B-Instruct"
set "FFMPEG_BIN=%ROOT_DIR%ffmpeg\bin"

if not exist "%ROOT_DIR%app.py" (
  echo MOSS-Audio has not been installed yet.
  echo Run install-audio-moss-standalone.bat first.
  exit /b 1
)

if not exist "%VENV_DIR%\Scripts\python.exe" (
  echo Virtual environment is missing.
  echo Run install-audio-moss-standalone.bat first.
  exit /b 1
)

if not exist "%WEIGHTS_DIR%" (
  echo Model weights are missing.
  echo Run install-audio-moss-standalone.bat first.
  exit /b 1
)

call "%VENV_DIR%\Scripts\activate.bat"
if errorlevel 1 exit /b 1

set "PATH=%FFMPEG_BIN%;%PATH%"
set "MOSS_AUDIO_MODEL_ID=%WEIGHTS_DIR%"
set "MOSS_AUDIO_SERVER_NAME=127.0.0.1"
set "MOSS_AUDIO_SERVER_PORT=7860"

python app.py
endlocal
