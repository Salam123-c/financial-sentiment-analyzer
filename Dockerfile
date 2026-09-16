# Multi-Stage Production Dockerfile for Institutional Quant NLP Engine (SR 11-7)
FROM python:3.11-slim AS builder

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

FROM python:3.11-slim
WORKDIR /app

# Copy installed wheels and dependencies
COPY --from=builder /root/.local /root/.local
COPY . /app

ENV PATH=/root/.local/bin:$PATH
ENV PYTHONUNBUFFERED=1

# Expose Streamlit (8501) and FastAPI (8000)
EXPOSE 8501
EXPOSE 8000

# Default entrypoint runs Streamlit Terminal
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
