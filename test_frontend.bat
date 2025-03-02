@echo off
echo HyperLiquid Frontend Test Script
echo ===================================

cd frontend
echo Installing dependencies...
call npm install
echo Starting development server...
call npm start
