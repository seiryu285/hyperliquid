@echo off
echo HyperLiquid Frontend Rebuild Script
echo ===================================

cd frontend
echo Installing dependencies...
call npm install
echo Building frontend...
call npm run build
echo Done!
pause
