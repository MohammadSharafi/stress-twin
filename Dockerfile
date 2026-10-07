# Container for Nebius Serverless Jobs: runs batches of Stress Twin episodes on CPU.
FROM python:3.11-slim

ENV MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
RUN apt-get update && apt-get install -y --no-install-recommends libosmesa6 libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
RUN pip install --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
# Prebuilt MuJoCo wheels only (building from source needs the C SDK). Pin MUJOCO_VERSION to
# the version used locally when you need cloud results to match local runs bit-for-bit.
ARG MUJOCO_VERSION=""
RUN pip install --no-cache-dir --only-binary=mujoco "mujoco${MUJOCO_VERSION:+==$MUJOCO_VERSION}"
COPY pyproject.toml README.md LICENSE ./
COPY stress_twin ./stress_twin
RUN pip install --no-cache-dir .

ENTRYPOINT []
CMD ["python", "-m", "stress_twin.worker", "--help"]
