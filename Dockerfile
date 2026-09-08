FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY . .

# Expose port (default 8000 or dynamic $PORT from cloud provider)
ENV PORT=8000
EXPOSE 8000

# Start FastAPI application
CMD ["python", "web_app.py"]
