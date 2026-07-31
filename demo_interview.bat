@echo off
setlocal EnableExtensions EnableDelayedExpansion
title Face Cluster Service - Live Demo
chcp 65001 >nul

echo =====================================================
echo   Face Cluster Service - Live Demo
echo   Hung Hing Printing / Interview take-home
echo =====================================================
echo.
cd /d "%~dp0"

echo [Step 1/4] Checking Docker containers...
docker ps --format "{{.Names}}" 2>nul | findstr /C:"face-cluster" >nul
if errorlevel 1 (
    echo   Containers not running. Starting stack...
    docker compose up -d
) else (
    echo   Containers already running.
)
echo.

echo [Step 2/4] Waiting for service to be ready...
echo   http://localhost:8765/health
set /a TRY=0
:healthloop
curl.exe -sf http://localhost:8765/health > _hc.tmp 2>nul
if %errorlevel%==0 goto healthup
set /a TRY+=1
if !TRY! GEQ 30 goto healthfail
echo   ...waiting (!TRY!s)...
timeout /t 2 /nobreak >nul
goto healthloop
:healthup
echo   Service is UP.
del _hc.tmp 2>nul
echo.

echo [Step 3/4] Clustering 9 sample images (Demo Mode)...
echo   POST /cluster  X-Demo-Mode:true  threshold=0.6
echo -------------------------------------------------
curl.exe -s -X POST http://localhost:8765/cluster ^
  -F "files=@tests\data\images\ident0_shot0.png" ^
  -F "files=@tests\data\images\ident0_shot1.png" ^
  -F "files=@tests\data\images\ident0_shot2.png" ^
  -F "files=@tests\data\images\ident1_shot0.png" ^
  -F "files=@tests\data\images\ident1_shot1.png" ^
  -F "files=@tests\data\images\ident1_shot2.png" ^
  -F "files=@tests\data\images\ident2_shot0.png" ^
  -F "files=@tests\data\images\ident2_shot1.png" ^
  -F "files=@tests\data\images\ident2_shot2.png" ^
  -H "X-Demo-Mode: true" -F "threshold=0.6"
echo.
echo -------------------------------------------------
echo.

echo [Step 4/4] Bonus: in-container end-to-end test...
docker exec face-cluster python /app/scripts/test_demo.py 2>nul
echo.

echo =====================================================
echo   Demo complete. Press any key to close.
echo =====================================================
pause >nul
exit /b 0

:healthfail
echo   ERROR: service not ready after ~60s.
echo   Run 'docker compose ps' to inspect.
del _hc.tmp 2>nul
pause
exit /b 1
