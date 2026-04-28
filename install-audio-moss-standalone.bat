@echo off
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"

set "ROOT_DIR=%~dp0"
set "VENV_DIR=%ROOT_DIR%.venv"
set "WEIGHTS_DIR=%ROOT_DIR%weights\MOSS-Audio-4B-Instruct"
set "HF_MODEL=OpenMOSS-Team/MOSS-Audio-4B-Instruct"
set "FFMPEG_DIR=%ROOT_DIR%ffmpeg"
set "FFMPEG_BIN=%FFMPEG_DIR%\bin"
set "DOWNLOADS_DIR=%ROOT_DIR%downloads"
set "FFMPEG_ZIP=%DOWNLOADS_DIR%\ffmpeg-release-essentials.zip"

where git >nul 2>&1
if errorlevel 1 (
  echo Git was not found on PATH.
  exit /b 1
)

where python >nul 2>&1
if errorlevel 1 (
  echo Python was not found on PATH.
  exit /b 1
)

if not exist "%DOWNLOADS_DIR%" mkdir "%DOWNLOADS_DIR%"

if not exist "%VENV_DIR%\Scripts\python.exe" (
  echo Creating virtual environment...
  python -m venv "%VENV_DIR%"
  if errorlevel 1 exit /b 1
)

call "%VENV_DIR%\Scripts\activate.bat"
if errorlevel 1 exit /b 1

python -m pip install --upgrade pip
if errorlevel 1 exit /b 1

if not exist "%FFMPEG_BIN%\ffmpeg.exe" (
  echo Downloading portable ffmpeg...
  curl -L -o "%FFMPEG_ZIP%" "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"
  if errorlevel 1 exit /b 1

  tar -xf "%FFMPEG_ZIP%" -C "%DOWNLOADS_DIR%"
  if errorlevel 1 exit /b 1

  set "FFMPEG_SOURCE="
  for /d %%D in ("%DOWNLOADS_DIR%\ffmpeg-*") do (
    if exist "%%~fD\bin\ffmpeg.exe" (
      set "FFMPEG_SOURCE=%%~fD"
    )
  )

  if not defined FFMPEG_SOURCE (
    echo Failed to locate ffmpeg after extraction.
    exit /b 1
  )

  if exist "%FFMPEG_DIR%" rmdir /s /q "%FFMPEG_DIR%"
  move "%FFMPEG_SOURCE%" "%FFMPEG_DIR%"
  if errorlevel 1 exit /b 1
)

set "PATH=%FFMPEG_BIN%;%PATH%"

echo Installing MOSS-Audio runtime...
pip install --extra-index-url https://download.pytorch.org/whl/cu128 -e ".[torch-runtime]"
if errorlevel 1 exit /b 1

echo Installing TorchCodec support...
pip install torchcodec
if errorlevel 1 exit /b 1

echo Installing YouTube download support...
pip install yt-dlp
if errorlevel 1 exit /b 1

echo Installing Hugging Face CLI support...
pip install "huggingface_hub[cli]"
if errorlevel 1 exit /b 1

if not exist "%WEIGHTS_DIR%" (
  echo Downloading model weights...
  hf download %HF_MODEL% --local-dir "%WEIGHTS_DIR%"
  if errorlevel 1 exit /b 1
)

echo.
echo Install complete.
echo Repo: %ROOT_DIR%
echo Model: %WEIGHTS_DIR%
echo Run: %ROOT_DIR%run-audio-moss-standalone.bat
endlocal
