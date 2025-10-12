# Multi-stage Docker build for Multi-Modal Recommendation Platform

# ============================================================================
# Stage 1: Base Image with Python and System Dependencies
# ============================================================================
FROM python:3.10-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    libgomp1 \
    libopenblas-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# ============================================================================
# Stage 2: Dependencies Installation
# ============================================================================
FROM base as dependencies

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install gunicorn uvicorn[standard]

# ============================================================================
# Stage 3: Application
# ============================================================================
FROM dependencies as application

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p \
    data/raw \
    data/processed \
    data/embeddings \
    checkpoints \
    logs

# Set permissions
RUN chmod +x scripts/*.py 2>/dev/null || true

# Expose ports
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["uvicorn", "src.api.server:app", "--host", "0.0.0.0", "--port", "8000"]

# ============================================================================
# Stage 4: Production Image (Optimized)
# ============================================================================
FROM application as production

# Remove development dependencies
RUN pip uninstall -y pytest pytest-cov black flake8 mypy

# Set production environment
ENV APP_ENV=production \
    DEBUG=false

# Use gunicorn for production
CMD ["gunicorn", "src.api.server:app", \
     "--workers", "4", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--bind", "0.0.0.0:8000", \
     "--timeout", "120", \
     "--access-logfile", "-", \
     "--error-logfile", "-"]

# ============================================================================
# Stage 5: Development Image
# ============================================================================
FROM application as development

# Install development dependencies
RUN pip install \
    ipython \
    jupyter \
    notebook \
    pytest-watch

# Set development environment
ENV APP_ENV=development \
    DEBUG=true \
    API_RELOAD=true

# Expose Jupyter port
EXPOSE 8888

# Development command with hot reload
CMD ["uvicorn", "src.api.server:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--reload"]
