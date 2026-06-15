# Shared image for both the FastAPI service and the Streamlit UI.
# docker-compose.yml runs the same image with two different commands.
#
# Data (data/), the prebuilt FAISS+BM25 index (data/index/), eval results (results/),
# and the HuggingFace model cache are intentionally NOT baked into this image -- they
# are externalized via volumes (see docker-compose.yml). This keeps the image small,
# decouples the data lifecycle from the code lifecycle, and honors the data/raw
# immutability rule (mounted read-only at runtime).
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.hf_cache

WORKDIR /app

# Install dependencies first (from pyproject.toml + src/) so this layer is cached
# independently of changes to configs/app/scripts below. Editable install is required:
# rag_eval.config resolves configs/config.yaml relative to the installed package source
# (src/rag_eval/config.py), so the source tree must remain at /app/src.
COPY pyproject.toml ./
COPY src/ src/
RUN pip install -e .

# App code that changes more often than the dependency set.
COPY configs/ configs/
COPY app.py ./
COPY scripts/ scripts/

# Non-root user with a writable model cache (bge embedding + reranker models download
# into HF_HOME on first use).
RUN useradd --create-home --shell /bin/bash app \
    && mkdir -p /app/.hf_cache \
    && chown -R app:app /app
USER app

EXPOSE 8000 8501

# Default command runs the API; docker-compose overrides this for the ui service.
CMD ["uvicorn", "rag_eval.api:app", "--host", "0.0.0.0", "--port", "8000"]
