# Step 1: Base Python 3.11 Slim Image
FROM python:3.11-slim

# Step 2: Set Working Directory inside container
WORKDIR /app

# Step 3: Install essential system dependencies (C compiler for binary packages)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Step 4: Copy and install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Step 5: Copy application source code
COPY . .

# Step 6: Expose port 8000 for FastAPI HTTP traffic
EXPOSE 8000

# Step 7: Container startup command using Uvicorn async ASGI server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
