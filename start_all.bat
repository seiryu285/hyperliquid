@echo off
echo HyperLiquid Trading System Starter
echo ===================================

echo Starting API server...
start cmd /k "cd api && python server.py"

echo Waiting for API server to start...
timeout /t 5

echo Starting frontend server...
cd frontend
call npm install
call npm start

echo All services started!
