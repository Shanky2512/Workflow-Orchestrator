@echo off
setlocal

set CONTAINER_NAME=echoai-memcached
set IMAGE=memcached:1.6.40-alpine
set PORT=11211
set MEMORY_MB=256

echo [1/3] Removing existing container if present...
docker rm -f %CONTAINER_NAME% >nul 2>nul

echo [2/3] Starting Memcached container...
docker run -d --name %CONTAINER_NAME% -p %PORT%:11211 %IMAGE% memcached -m %MEMORY_MB% -p 11211 -v
if errorlevel 1 (
  echo Failed to start Memcached container.
  exit /b 1
)

echo [3/3] Memcached is up on localhost:%PORT%
echo Suggested .env values:
echo   MEMCACHED_ENABLED=true
echo   MEMCACHED_HOSTS=localhost:%PORT%
echo   MEMCACHED_TTL=1800
echo   MEMCACHED_FALLBACK=true

endlocal
