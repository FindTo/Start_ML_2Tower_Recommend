# Python 3.11 slim
FROM python:3.11-slim

# Working directory
WORKDIR /app/sources

# Copy requirements separetely (for caching)
COPY requirements.txt .

# Upgrade Pip
RUN pip install --no-cache-dir --upgrade pip

# Install dependencies - no Torch
RUN pip install --no-cache-dir $(grep -v "^torch" requirements.txt)

# Install Torch CPU version for server
RUN pip install --no-cache-dir  torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu

# Copy sources files - scripts and models
COPY sources/ ./

# Command to launch web server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]