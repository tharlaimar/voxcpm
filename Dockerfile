FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /workspace

# Python 3.12 install
RUN apt-get update && apt-get install -y \
    ffmpeg \
    python3.12 \
    python3.12-dev \
    python3.12-distutils \
    && rm -rf /var/lib/apt/lists/*

# Python 3.12 ကို default လုပ်
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12

COPY requirements.txt .
RUN python3.12 -m pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python3.12", "-u", "app_api.py"]
