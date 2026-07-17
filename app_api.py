import os
import gc
import re
import requests
import base64
import numpy as np
import soundfile as sf
import torch

import torch._dynamo
torch._dynamo.config.disable = True

import runpod
from voxcpm import VoxCPM

# ================================================================
# GPU Check
# ================================================================
# GPU မရှိရင် worker ချက်ချင်း exit
if not torch.cuda.is_available():
    raise RuntimeError("[FATAL] CUDA GPU required but not found!")

# GPU ကို default device အဖြစ် သတ်မှတ်
torch.set_default_device("cuda")
device = torch.device("cuda")
print(f"[INIT] Using GPU: {torch.cuda.get_device_name(0)}")

# ================================================================
# Model Global Load
# ================================================================
MODEL_PATH = "/runpod-volume/VoxCPM2"

print(f"[INIT] Loading VoxCPM2 from {MODEL_PATH} ...")
model = VoxCPM.from_pretrained(MODEL_PATH, load_denoiser=False, local_files_only=True)
print("[INIT] Model loaded successfully!")


# ================================================================
# မြန်မာစာ Sentence Splitting
# ================================================================
def split_myanmar_text(text: str, max_chars: int = 80) -> list[str]:
    # style tag တွေ (ကွင်းစကွင်းပိတ်) ဖယ်ရှားတယ်
    clean = re.sub(r'\[.*?\]', '', text)
    clean = re.sub(r'\(.*?\)', '', clean)

    # sentence boundary တွေမှာ ဖြတ်တယ်
    smart = (clean
             .replace('။', '။\n')
             .replace('.', '.\n')
             .replace('?', '?\n')
             .replace('!', '!\n'))

    sentences = [t.strip() for t in smart.split('\n') if t.strip()]

    chunks = []
    current = ""
    for sentence in sentences:
        if len(sentence) > max_chars:
            if current:
                chunks.append(current.strip())
                current = ""
            for i in range(0, len(sentence), max_chars):
                chunks.append(sentence[i:i + max_chars].strip())
            continue
        if len(current) + len(sentence) <= max_chars:
            current += sentence
        else:
            if current:
                chunks.append(current.strip())
            current = sentence
    if current:
        chunks.append(current.strip())

    return [c for c in chunks if c]


# ================================================================
# Chunked Generation — VRAM safe
# ================================================================
def generate_chunked(text: str, **kwargs) -> tuple[np.ndarray, int]:
    chunks = split_myanmar_text(text)
    print(f"[GEN] {len(chunks)} chunks: {[len(c) for c in chunks]} chars")

    # model ရဲ့ actual sample rate ယူတယ်
    actual_sr = model.tts_model.sample_rate
    silence   = np.zeros(int(actual_sr * 0.5), dtype=np.float32)

    # VRAM ကို generation မတိုင်ခင် clear လုပ်တယ်
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    # quality settings — VRAM safe
    kwargs['cfg_value']           = 2.0
    kwargs['inference_timesteps'] = 10

    audio_parts = []

    for i, chunk in enumerate(chunks):
        if len(chunk.strip()) < 2:
            continue

        print(f"[GEN] chunk {i+1}/{len(chunks)}: {chunk[:50]}...")

        with torch.inference_mode():
            wav = model.generate(text=chunk + " ", **kwargs)

            # tensor → numpy
            if isinstance(wav, tuple):
                wav = wav[0]
            if isinstance(wav, torch.Tensor):
                wav = wav.detach().cpu().numpy()

            wav = wav.astype(np.float32).flatten()
            audio_parts.append(wav)

            if i < len(chunks) - 1:
                audio_parts.append(silence)

        # chunk တစ်ခုပြီးတိုင်း memory clean လုပ်တယ်
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    if not audio_parts:
        return np.zeros(100, dtype=np.float32), actual_sr

    return np.concatenate(audio_parts), actual_sr


# ================================================================
# Helper functions
# ================================================================
def encode_audio(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def download_file(url: str, dest: str) -> None:
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    with open(dest, "wb") as f:
        f.write(r.content)


# ================================================================
# Handler
#
# ── Mode 1: style ──────────────────────────────────────────────
#   Voice style prompt နဲ့ generate လုပ်တာ — reference audio မလို
#
#   {
#     "input": {
#       "action": "style",
#       "text":   "မင်္ဂလာပါ။ ကျွန်တော် VoxCPM ကို သုံးနေပါတယ်။",
#       "style":  "A warm female voice, gentle and calm"
#     }
#   }
#
# ── Mode 2: preset ─────────────────────────────────────────────
#   Frontend မှာ သိမ်းထားတဲ့ နမူနာ audio URL ကို reference သုံးတာ
#
#   {
#     "input": {
#       "action":         "preset",
#       "text":           "မင်္ဂလာပါ။ ကျွန်တော် VoxCPM ကို သုံးနေပါတယ်။",
#       "audio_url":      "https://your-cdn.com/voices/sample1.wav",
#       "reference_text": "preset audio ထဲက စာသား"   ← optional
#     }
#   }
#
# ── Mode 3: clone ──────────────────────────────────────────────
#   User upload လုပ်တဲ့ audio + reference text နဲ့ အသံ clone လုပ်တာ
#
#   {
#     "input": {
#       "action":         "clone",
#       "text":           "ထုတ်ချင်တဲ့ စာသား",
#       "audio_url":      "https://your-storage.com/user-upload.wav",
#       "reference_text": "reference audio ထဲက စာသား"   ← ထည့်ရင် quality ပိုကောင်း
#     }
#   }
#
# ── Output ────────────────────────────────────────────────────
#   {
#     "status":       "success",
#     "audio_base64": "...",
#     "sample_rate":  48000
#   }
# ================================================================
def handler(job):
    job_input = job["input"]
    action    = job_input.get("action", "style")
    text      = job_input.get("text", "မင်္ဂလာပါ။")
    out_path  = "/tmp/output.wav"

    gen_kwargs = {}

    try:

        # ── Mode 1: Style (+ "design" backward compat) ────────────
        if action in ["style", "design"]:
            style     = job_input.get("style", "")
            full_text = f"({style}){text}" if style else text
            final_wav, actual_sr = generate_chunked(full_text, **gen_kwargs)
            sf.write(out_path, final_wav, actual_sr)

        # ── Mode 2: Preset ────────────────────────────────────────
        elif action == "preset":
            audio_url      = job_input.get("audio_url")
            reference_text = job_input.get("reference_text", "").strip()
            if not audio_url:
                return {"status": "error", "message": "audio_url is required for preset"}

            ref_path = "/tmp/preset_ref.wav"
            download_file(audio_url, ref_path)

            gen_kwargs["prompt_wav_path"] = ref_path
            if reference_text:
                gen_kwargs["prompt_text"] = reference_text

            final_wav, actual_sr = generate_chunked(text, **gen_kwargs)
            sf.write(out_path, final_wav, actual_sr)

        # ── Mode 3: Clone ─────────────────────────────────────────
        elif action == "clone":
            audio_url      = job_input.get("audio_url")
            reference_text = job_input.get("reference_text", "").strip()
            if not audio_url:
                return {"status": "error", "message": "audio_url is required for clone"}

            ref_path = "/tmp/clone_ref.wav"
            download_file(audio_url, ref_path)

            gen_kwargs["prompt_wav_path"] = ref_path
            if reference_text:
                gen_kwargs["prompt_text"] = reference_text

            final_wav, actual_sr = generate_chunked(text, **gen_kwargs)
            sf.write(out_path, final_wav, actual_sr)

        else:
            return {
                "status":  "error",
                "message": f"Unknown action '{action}'. Use 'style', 'preset', or 'clone'."
            }

        # ── Response ──────────────────────────────────────────────
        if os.path.exists(out_path):
            return {
                "status":       "success",
                "audio_base64": encode_audio(out_path),
                "sample_rate":  actual_sr,
            }
        else:
            return {"status": "error", "message": "Audio file was not created"}

    except Exception as e:
        import traceback
        return {
            "status":    "error",
            "message":   str(e),
            "traceback": traceback.format_exc(),
        }


# ================================================================
if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
