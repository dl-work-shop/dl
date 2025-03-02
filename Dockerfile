# Stage 1: Build

FROM python:3.9-slim as builder

# Install system dependencies
RUN apt-get update \
    apt-get install -y --no-install-recommends gcc python3-dev && \
    rm -rf /var/lib/apt/lists

# Install Python dependencies
COPY requirements_production.txt .
RUN pip install --user -r requirements_production.txt

# Stage 2: Final

FROM python:3.9-slim

WORKDIR /app

# Copy Python dependencies
COPY --from=builder /root/.local /root/.local
COPY . .

# Set environment variables
ENV PATH=/root/.local/bin:$PATH
ENV STREAMLIT_SERVER_PORT=8501

# Expose port
EXPOSE 8501

# Run the application
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]

