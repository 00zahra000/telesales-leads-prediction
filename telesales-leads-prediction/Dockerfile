FROM python:3.12-slim AS builder

WORKDIR /app

COPY requirements.txt .

RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install -r requirements.txt


FROM python:3.12-slim AS runtime

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH"

COPY --from=builder /opt/venv /opt/venv
COPY src/ ./src/
COPY tests/ ./tests/

CMD ["python", "-m", "src.pipeline"]
