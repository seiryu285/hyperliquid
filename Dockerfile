# Multi-stage build for optimized production image
FROM node:18-alpine as frontend-build

# Set working directory
WORKDIR /app/frontend

# Copy frontend dependencies
COPY frontend/package*.json ./

# Install dependencies
RUN npm ci

# Copy frontend source
COPY frontend/src ./src
COPY frontend/public ./public
COPY frontend/tsconfig.json .

# Build frontend
RUN npm run build

# Python backend stage
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy backend requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend source
COPY backend ./backend
COPY simulator ./simulator
COPY risk_management ./risk_management
COPY config ./config

# Copy frontend build from previous stage
COPY --from=frontend-build /app/frontend/build ./frontend/build

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV NODE_ENV=production

# Expose ports
EXPOSE 8000

# Create volume for data
VOLUME ["/app/data"]

# Start command
CMD ["python", "-m", "backend.websocket_server"]
