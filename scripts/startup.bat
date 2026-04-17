:: =============================================================================
:: Author : B.Vignesh Kumar aka Bravetux <ic19939@gmail.com>
:: Date   : 26 March 2026
:: =============================================================================
@echo off
setlocal enabledelayedexpansion
title TechTrainer AI — Startup

:: ── Change to project root (one level up from scripts\) ──────
cd /d "%~dp0.."

echo ============================================================
echo   TechTrainer AI — Startup Check
echo ============================================================
echo.

:: ── 1. Check Python ──────────────────────────────────────────
echo [1/6] Checking Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo   [FAIL] Python not found. Please install Python 3.10+ and add it to PATH.
    echo          Download from: https://www.python.org/downloads/
    goto :error
)
for /f "tokens=2" %%v in ('python --version 2^>^&1') do set PY_VER=%%v
echo   [OK]   Python %PY_VER% found.

:: ── 2. Check uv ──────────────────────────────────────────────
echo [2/6] Checking uv package manager...
uv --version >nul 2>&1
if errorlevel 1 (
    echo   [WARN] uv not found. Installing uv now...
    pip install uv >nul 2>&1
    if errorlevel 1 (
        echo   [FAIL] Could not install uv. Install manually:
        echo          pip install uv
        echo          OR: https://docs.astral.sh/uv/getting-started/installation/
        goto :error
    )
    echo   [OK]   uv installed.
) else (
    for /f "tokens=1,2" %%a in ('uv --version 2^>^&1') do set UV_VER=%%b
    echo   [OK]   uv !UV_VER! found.
)

:: ── 3. Check .env ────────────────────────────────────────────
echo [3/6] Checking .env configuration...
if not exist ".env" (
    echo   [WARN] .env file not found.
    if exist ".env.example" (
        echo          Copying .env.example to .env...
        copy ".env.example" ".env" >nul
        echo   [WARN] .env created from template. Please edit it with your AWS credentials:
        echo          AWS_ACCESS_KEY_ID=your_key_here
        echo          AWS_SECRET_ACCESS_KEY=your_secret_here
        echo.
        set /p OPEN_ENV="   Open .env in Notepad now? [Y/N]: "
        if /i "!OPEN_ENV!"=="Y" (
            notepad .env
            echo   Waiting for you to save .env...
            pause
        )
    ) else (
        echo   [FAIL] Neither .env nor .env.example found. Cannot proceed.
        goto :error
    )
) else (
    :: Check that AWS keys are not placeholder values
    findstr /i "your_key_here\|your_secret_here\|CHANGE_ME" .env >nul 2>&1
    if not errorlevel 1 (
        echo   [WARN] .env contains placeholder values. Please update your AWS credentials.
        set /p OPEN_ENV2="   Open .env in Notepad now? [Y/N]: "
        if /i "!OPEN_ENV2!"=="Y" (
            notepad .env
            echo   Waiting for you to save .env...
            pause
        )
    ) else (
        echo   [OK]   .env found and configured.
    )
)

:: ── 4. Install / sync dependencies ───────────────────────────
echo [4/6] Syncing Python dependencies with uv...
uv sync --quiet
if errorlevel 1 (
    echo   [FAIL] Dependency sync failed. Check pyproject.toml and internet connectivity.
    echo          Try manually: uv sync
    goto :error
)
echo   [OK]   All dependencies installed.

:: ── 5. Check / run document ingestion ────────────────────────
echo [5/6] Checking knowledge base index...
set DATA_DIR=data\documents

if not exist "%DATA_DIR%" (
    echo   [INFO] Place training documents in data\documents\ and re-run to index them.
    echo          Supported: .pdf .docx .pptx .xlsx .txt .md
    goto :launch
)

:: Count files in documents directory
set FILE_COUNT=0
for %%f in ("%DATA_DIR%\*") do set /a FILE_COUNT+=1

if %FILE_COUNT%==0 (
    echo   [INFO] No documents in data\documents\ yet. Add training files and re-run.
    echo          Supported: .pdf .docx .pptx .xlsx .txt .md
    goto :launch
)

echo   [INFO] Found documents. Running ingestion check...
uv run python -m src.tools.document_ingestion --status >nul 2>&1
if errorlevel 1 (
    echo   [INFO] Indexing %FILE_COUNT% document(s) into knowledge base...
    uv run python -m src.tools.document_ingestion
    if errorlevel 1 (
        echo   [WARN] Ingestion encountered errors. App will still start.
    ) else (
        echo   [OK]   Knowledge base indexed successfully.
    )
) else (
    echo   [OK]   Knowledge base index already up to date.
)

:launch
:: ── 6. Launch app ────────────────────────────────────────────
echo [6/6] Launching TechTrainer AI...
echo.
echo ============================================================
echo   App starting at: http://localhost:8502
echo   Press Ctrl+C in this window to stop the server.
echo ============================================================
echo.

:: Open browser after a short delay (let Streamlit boot first)
start "" /b cmd /c "timeout /t 3 >nul && start http://localhost:8502"

:: Start Streamlit
uv run streamlit run app.py --server.port 8502 --server.headless false
goto :eof

:error
echo.
echo ============================================================
echo   Startup failed. Fix the issues above and re-run startup.bat
echo ============================================================
pause
exit /b 1
