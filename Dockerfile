# PyTorch 2.6 + CUDA 12.4.0 — driver 12.4.0 နဲ့ exact match
# VoxCPM2 requires PyTorch >= 2.5.0
FROM runpod/pytorch:2.6.0-py3.11-cuda12.4.0-devel-ubuntu22.04

WORKDIR /workspace

# အသံဖိုင်တွေ လုပ်ဆောင်ဖို့အတွက် လိုအပ်တဲ့ ffmpeg စနစ်ကို သွင်းခြင်း
RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

# requirements.txt ကို အရင်ကူးပြီး library တွေ သွင်းမယ်
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ကျန်တဲ့ ကုဒ်တွေအကုန်လုံးကို Image ထဲ ကူးထည့်မယ်
COPY . .

# Serverless စမောင်းတာနဲ့ app_api.py ကို တန်း Run မယ်
CMD ["python", "-u", "app_api.py"]
