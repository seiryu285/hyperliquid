@echo off
echo HyperLiquid API Server Starter
echo ===================================

echo Installing dependencies...
pip install -r requirements.txt
echo Starting API server...
cd api
python server.py
