FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PROJECT_ROOT=/app \
    MPLCONFIGDIR=/tmp/matplotlib

RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        tesseract-ocr \
        tesseract-ocr-rus \
        tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt pyproject.toml README.md LICENSE ./
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY scripts ./scripts
COPY models ./models
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh

RUN pip install --no-cache-dir --no-deps . \
    && chmod +x /usr/local/bin/entrypoint.sh \
    && useradd --create-home --uid 1000 detector

FROM base AS runtime

COPY data/datasets ./data/datasets

RUN mkdir -p data/raw/drweb data/interim data/processed reports/metrics \
    && chown -R detector:detector /app

USER detector

ENTRYPOINT ["entrypoint.sh"]
CMD ["analyze", "--input", "data/raw/drweb", "--scorer", "heuristic"]

FROM base AS metrics

COPY data/datasets ./data/datasets
COPY tests ./tests

RUN mkdir -p data/interim reports/metrics \
    && chown -R detector:detector /app

USER detector

ENTRYPOINT ["entrypoint.sh"]
CMD ["evaluate", "--split", "test", "--scorer", "heuristic"]
