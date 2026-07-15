import runpod
import os
import re
import requests
import base64
import numpy as np
import soundfile as sf
import sys
import torch

# 🌟 [FAIL-FAST SYSTEM] Driver အဟောင်းကြောင့်ဖြစ်စေ၊ GPU မမိလို့ဖြစ်စေ CPU ပေါ်ရောက်သွားရင် တန်းကစ်မယ်
if not torch.cuda.is_available():
    print("[CRITICAL] CUDA/GPU မမိပါ! (Driver အဟောင်းဖြစ်နေနိုင်သည်)")
    print("[CRITICAL] Worker ကို ချက်ချင်း ပိတ်ချပြီး Job ကို Failed အဖြစ် သတ်မှတ်ပါမည်။")
    sys.exit(1)
from voxcpm import VoxCPM

# ================================================================
# Model Global Load — worker start တဲ့အချိန် တစ်ကြိမ်ပဲ run မယ်
# ================================================================
MODEL_PATH  = "/runpod-volume/VoxCPM2"
SAMPLE_RATE = 48000   # VoxCPM2 native output
MAX_CHARS   = 150     # chunk တစ်ခုရဲ့ max character (RAM safe)

print(f"[INIT] Loading VoxCPM2 from {MODEL_PATH} ...")
model = VoxCPM.from_pretrained(MODEL_PATH, load_denoiser=False)
print("[INIT] Model loaded successfully!")


# ================================================================
# မြန်မာစာ Chunking
# ================================================================
def split_myanmar_text(text: str, max_chars: int = MAX_CHARS) -> list[str]:
    """
    မြန်မာစာကို ။ နဲ့ ၊ နဲ့ ဖြတ်ပြီး chunk တွေ ခွဲတယ်။
    chunk တစ်ခု max_chars ထက် မကျော်ဘဲ တတ်နိုင်သမျှ ပေါင်းသိမ်းတယ်။
    """
    # ။ နဲ့ ၊ မှာ ဖြတ်တယ် — delimiter ကို sentence နောက်မှာ ထားတယ်
    sentences = re.split(r'(?<=[။၊])', text)
    sentences = [s.strip() for s in sentences if s.strip()]

    chunks = []
    current = ""

    for sentence in sentences:
        # sentence တစ်ခုထဲကိုယ်တိုင် max_chars ထက် ကျော်နေရင် hard split လုပ်
        if len(sentence) > max_chars:
            if current:
                chunks.append(current.strip())
                current = ""
            # space နဲ့ hard split
            for i in range(0, len(sentence), max_chars):
                chunks.append(sentence[i:i+max_chars].strip())
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


def generate_chunked(text: str, **kwargs) -> np.ndarray:
    """
    Long text ကို chunk တွေ ခွဲပြီး generate လုပ်ပြီး concatenate လုပ်တယ်။
    RAM overflow မဖြစ်အောင် တစ်ခုပြီး တစ်ခု generate လုပ်တယ်။
    """
    chunks = split_myanmar_text(text)
    print(f"[CHUNK] Split into {len(chunks)} chunks: {[len(c) for c in chunks]} chars")

    audio_parts = []
    for i, chunk in enumerate(chunks):
        print(f"[CHUNK] Generating chunk {i+1}/{len(chunks)}: {chunk[:40]}...")
        wav = model.generate(text=chunk, **kwargs)
        audio_parts.append(wav)

    if len(audio_parts) == 1:
        return audio_parts[0]

    # chunk တွေ ကြား short silence (0.2s) ထည့်ပြီး join လုပ်တယ်
    silence = np.zeros(int(SAMPLE_RATE * 0.2), dtype=np.float32)
    combined = audio_parts[0]
    for part in audio_parts[1:]:
        combined = np.concatenate([combined, silence, part])

    return combined


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
#   Prompt နဲ့ voice style ပြောင်းတာ — reference audio မလို
#
#   {
#     "input": {
#       "action": "style",
#       "text":   "မင်္ဂလာပါ။ ကျွန်တော် VoxCPM ကို သုံးနေပါတယ်။",
#       "style":  "A warm female voice, gentle and calm"   ← optional
#     }
#   }
#
# ── Mode 2: preset ─────────────────────────────────────────────
#   Frontend မှာ သိမ်းထားတဲ့ နမူနာ audio URL ကို reference သုံးတာ
#
#   {
#     "input": {
#       "action":    "preset",
#       "text":      "မင်္ဂလာပါ။ ကျွန်တော် VoxCPM ကို သုံးနေပါတယ်။",
#       "audio_url": "https://your-cdn.com/voices/sample1.wav"
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
#       "reference_text": "reference audio ထဲက စာသား"   ← optional
#     }
#   }
#
# ── Output (သုံးမျိုးလုံး) ────────────────────────────────────
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

    gen_kwargs = dict(
        cfg_value=2.0,
        inference_timesteps=10,
    )

    try:

        # ── Mode 1: Style ─────────────────────────────────────────
        if action == "style":
            style     = job_input.get("style", "")
            full_text = f"({style}){text}" if style else text

            wav = generate_chunked(full_text, **gen_kwargs)
            sf.write(out_path, wav, SAMPLE_RATE)

        # ── Mode 2: Preset ────────────────────────────────────────
        elif action == "preset":
            audio_url = job_input.get("audio_url")
            if not audio_url:
                return {"status": "error", "message": "audio_url is required for preset action"}

            ref_path = "/tmp/preset_ref.wav"
            download_file(audio_url, ref_path)

            wav = generate_chunked(
                text,
                prompt_wav_path=ref_path,
                **gen_kwargs,
            )
            sf.write(out_path, wav, SAMPLE_RATE)

        # ── Mode 3: Clone ─────────────────────────────────────────
        elif action == "clone":
            audio_url      = job_input.get("audio_url")
            reference_text = job_input.get("reference_text", "")

            if not audio_url:
                return {"status": "error", "message": "audio_url is required for clone action"}

            ref_path = "/tmp/clone_ref.wav"
            download_file(audio_url, ref_path)

            if reference_text:
                # Ultimate cloning — timbre + prosody + nuance အကုန် ကူးတယ်
                wav = generate_chunked(
                    text,
                    
                    prompt_wav_path=ref_path,
                    prompt_text=reference_text,
                    **gen_kwargs,
                )
            else:
                # Controllable cloning — timbre ပဲ ကူးတယ်
                wav = generate_chunked(
                    text,
                    
                    **gen_kwargs,
                )
            sf.write(out_path, wav, SAMPLE_RATE)

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
                "sample_rate":  SAMPLE_RATE,
            }
        else:
            return {"status": "error", "message": "Audio file was not created"}

    except Exception as e:
        import traceback
        return {"status": "error", "message": str(e), "traceback": traceback.format_exc()}


# ================================================================
if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
