FROM python:3.11-slim

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK stopwords at build time
RUN python -c "import nltk; nltk.download('stopwords', quiet=True)"

# Copy source
COPY . .

# Default: run FastAPI API
CMD ["uvicorn", "service.api:app", "--host", "0.0.0.0", "--port", "8000"]
