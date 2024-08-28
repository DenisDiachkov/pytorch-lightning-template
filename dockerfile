FROM python:3.11-slim

COPY dist /dist

RUN apt-get update && apt-get install -y \
    libpq-dev \
    libgl1-mesa-glx \
    libglib2.0-0 libsm6 libxrender1 libxext6 \
    gcc \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /root/pip
RUN --mount=type=secret,id=PIP_INDEX_EXTRA_URL,target=/PIP_INDEX_EXTRA_URL \
    python -m pip install --upgrade pip && \
    python -m pip install /dist/*.whl --no-cache-dir \
    --extra-index-url $(cat /PIP_INDEX_EXTRA_URL) \
    --trusted-host $(cat /PIP_INDEX_EXTRA_URL  | awk -F/ '{print $3}' | awk -F@ '{print $2}')

RUN echo $(ls /dist/*.whl | sed 's/.*\///' | sed 's/-.*//') > /PACKAGE_NAME
ENTRYPOINT  python -m $(cat /PACKAGE_NAME) --cfg /cfg/config.yaml