FROM python:3.9.6-slim-bullseye

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        git \
        liblapack-dev \
        libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

ENV OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r /app/requirements.txt

COPY . /app

RUN python -c "import numpy, scipy, pandas, sklearn, networkx, statsmodels, arch, community; print('environment ok')"

CMD ["python", "run_pipeline.py"]
