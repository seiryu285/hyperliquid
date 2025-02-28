# Project Overview

This project is designed for the Hyper Liquid Agent, including environment setup, data processing, execution strategies, risk management, RL agents, and more.

## Setup
Run the setup_project.py script to generate the directory structure.

## CI/CD Pipeline

The project includes a GitHub Actions workflow for continuous integration and continuous deployment. The workflow is configured to:

1. Run tests on every push to the main, master, develop, and feature branches
2. Build and push a Docker image on pushes to the main or master branch
3. Deploy the application to the production environment on pushes to the main or master branch

The CI/CD pipeline is defined in `.github/workflows/ci-cd.yml`.

## Data Visualization

The project includes a data visualization dashboard for monitoring market data from HyperLiquid. The dashboard provides:

- Real-time price charts with candlestick data
- Order book visualization
- Technical indicators (RSI, MACD, Bollinger Bands, etc.)
- Position summary and risk assessment
- Recent trades table

### Frontend

The frontend is built with React and TypeScript, using Material-UI for the component library. The main dashboard components are located in `frontend/src/pages/Dashboard/`.

### API

The backend API is built with FastAPI and provides endpoints for:

- Market data (order book, trades, ticker)
- OHLCV data with technical indicators
- Account information
- Order management (create, cancel, history)

The API server is defined in `api/server.py`.

## Running the Application

1. Create a `.env` file based on `.env.example` with your HyperLiquid API credentials
2. Start the backend API:
   ```
   cd project_root
   python -m api.server
   ```
3. Start the frontend development server:
   ```
   cd project_root/frontend
   npm install
   npm start
   ```
4. Access the dashboard at http://localhost:3000