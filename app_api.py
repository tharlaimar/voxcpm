import sys
print(f"[DEBUG] Python: {sys.version}", flush=True)

import torch
print(f"[DEBUG] torch: {torch.__version__}", flush=True)

import torchaudio
print(f"[DEBUG] torchaudio: {torchaudio.__version__}", flush=True)

import torchcodec
print(f"[DEBUG] torchcodec OK", flush=True)

from voxcpm import VoxCPM
print(f"[DEBUG] voxcpm OK", flush=True)

import runpod
print(f"[DEBUG] runpod OK", flush=True)

def handler(job):
    return {"status": "ok"}

runpod.serverless.start({"handler": handler})
