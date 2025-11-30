# Stage 1: Build the Application
FROM python:3.11 AS build

WORKDIR /usr/src/app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements.txt
COPY requirements.txt ./requirements.txt

# Install Python dependencies
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy app source
COPY . .

# Stage 2: Runtime image
FROM python:3.11

WORKDIR /usr/src/app

# Copy venv and app
COPY --from=build /opt/venv /opt/venv
COPY --from=build /usr/src/app .

ENV PATH="/opt/venv/bin:$PATH"

# Non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /usr/src/app
USER appuser

ENV PORT=8080
EXPOSE $PORT

# Start app
CMD ["python", "app.py"]
