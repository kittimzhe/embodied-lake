FROM python:3.12-slim

# embodied-lake self-hosted deployment: API + catalog over a volume mount.
# Built from the repo root: docker build -f docker/Dockerfile -t embodied-lake .
RUN pip install --no-cache-dir pyarrow pandas numpy mcap mcap-ros2-support h5py rerun-sdk

WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir --no-deps .
ENV PYTHONPATH=/app/src

# datasets + catalog live on mounted volumes
VOLUME ["/data", "/var/lib/elake"]

ENV ELAKE_CATALOG=/var/lib/elake
EXPOSE 8317

CMD ["python", "-m", "embodied_lake", "serve", "--catalog", "/var/lib/elake", "--host", "0.0.0.0", "--port", "8317"]
