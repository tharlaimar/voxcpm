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
MAX_CHARS   = 150     # chunk တစ်ခုရဲ့ max character (RAM safe)

print(f"[INIT] Loading VoxCPM2 from {MODEL_PATH} ...")
model = VoxCPM.from_pretrained(MODEL_PATH, load_denoiser=False)
print("[INIT] Model loaded successfully!")


# ================================================================
# Text Chunking (မြန်မာ/အင်္ဂလိပ်)
# ================================================================
def split_myanmar_text(text: str, max_chars: int = MAX_CHARS) -> list[str]:
    """
    မြန်မာစာကို (။၊) နဲ့ အင်္ဂလိပ်စာကို (.,) နဲ့ ဖြတ်ပြီး chunk တွေ ခွဲတယ်။
    chunk တစ်ခု max_chars ထက် မကျော်ဘဲ တတ်နိုင်သမျှ ပေါင်းသိမ်းတယ်။
    """
    # 💡 အင်္ဂလိပ်စာလုံးတွေ အလယ်ကနေ အတင်းမဖြတ်မိအောင် Full stop (.) နဲ့ Comma (,) ပါ ထည့်ထားပါတယ်
    sentences = re.split(r'(?<=[။၊.,])', text)
    sentences = [s.strip() for s in sentences if s.strip()]

    chunks = []
    current = ""

    for sentence in sentences:
        if len(sentence) > max_chars:
            if current:
                chunks.append(current.strip())
                current = ""
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


# ================================================================
# Generation System
# ================================================================
def generate_chunked(text: str, **kwargs) -> tuple[np.ndarray, int]:
    """
    Long text ကို chunk တွေ ခွဲပြီး generate လုပ်ပြီး concatenate လုပ်တယ်။
    """
    chunks = split_myanmar_text(text)
    print(f"[CHUNK] Split into {len(chunks)} chunks: {[len(c) for c in chunks]} chars")

    # 💡 VoxCPM ရဲ့ မှန်ကန်တဲ့ Sample Rate ကို ယူပါမယ်
    try:
        actual_sr = int(model.tts_model.sample_rate)
    except:
        actual_sr = 24000 # အရန်အနေနဲ့ ၂၄၀၀၀ ထားပါမယ်
        
    silence_len = int(actual_sr * 0.15)
    silence = np.zeros(silence_len, dtype=np.float32)

    audio_parts = []
    
    for i, chunk in enumerate(chunks):
        if len(chunk.strip()) < 2:
            continue
            
        print(f"[CHUNK] Generating chunk {i+1}/{len(chunks)}...")

        with torch.inference_mode():
            safe_text = chunk + " "
            wav = model.generate(text=safe_text, **kwargs)
            
            if isinstance(wav, torch.Tensor):
                wav = wav.detach().cpu().numpy()
            wav = np.array(wav).astype(np.float32).flatten()
            
            audio_parts.append(wav)
            if i < len(chunks) - 1:
                audio_parts.append(silence)
                
        torch.cuda.empty_cache()

    if not audio_parts:
        return np.zeros(100, dtype=np.int16), actual_sr

    # အပိုင်းတွေကို ပေါင်းပါမယ်
    combined = np.concatenate(audio_parts)

    # 🚀 [အရေးကြီးဆုံး ပြင်ဆင်ချက်: Audio Normalization] 🚀
    # အသံလှိုင်း (Waveform) က Volume အရမ်းကျယ်နေပြီး Clipping ဖြစ်နေတာကို 
    # ပုံမှန်အသံဖြစ်အောင် အချိုးကျ ပြန်ချုံ့ပေးပါမယ် (Normalize)
    max_val = np.max(np.abs(combined))
    if max_val > 0:
        combined = combined / max_val  # အသံကို -1.0 နှင့် 1.0 ကြားရောက်အောင် ညှိပါမည်
        
    # Flutter က အကောင်းဆုံး နားလည်တဲ့ Standard 16-bit PCM (WAV) format သို့ ပြောင်းပါမည်
    combined = (combined * 32767).astype(np.int16)

    return combined, actual_sr

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

            wav, actual_sr = generate_chunked(full_text, **gen_kwargs)
            sf.write(out_path, wav, actual_sr)

        # ── Mode 2: Preset ────────────────────────────────────────
        elif action == "preset":
            audio_url = job_input.get("audio_url")
            if not audio_url:
                return {"status": "error", "message": "audio_url is required for preset action"}

            ref_path = "/tmp/preset_ref.wav"
            download_file(audio_url, ref_path)

            # 💡 prompt_text မရှိလျှင် prompt_wav_path မထည့်ပါ (library error ရှောင်ရန်)
            wav, actual_sr = generate_chunked(
                text,
                **gen_kwargs,
            )
            sf.write(out_path, wav, actual_sr)

        # ── Mode 3: Clone ─────────────────────────────────────────
        elif action == "clone":
            audio_url      = job_input.get("audio_url")
            reference_text = job_input.get("reference_text", "")

            if not audio_url:
                return {"status": "error", "message": "audio_url is required for clone action"}

            ref_path = "/tmp/clone_ref.wav"
            download_file(audio_url, ref_path)

            if reference_text:
                # 💡 နှစ်ခုလုံးရှိလျှင် တွဲထည့်ပါမည်
                wav, actual_sr = generate_chunked(
                    text,
                    prompt_wav_path=ref_path,
                    prompt_text=reference_text,
                    **gen_kwargs,
                )
            else:
                # 💡 prompt_text မရှိလျှင် ဘာမှမထည့်ပါ
                wav, actual_sr = generate_chunked(
                    text,
                    **gen_kwargs,
                )
            sf.write(out_path, wav, actual_sr)

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
                "sample_rate":  actual_sr, # 💡 မှန်ကန်သော sample rate ကို ပြန်ပို့ပါမည်
            }
        else:
            return {"status": "error", "message": "Audio file was not created"}

    except Exception as e:
        import traceback
        return {"status": "error", "message": str(e), "traceback": traceback.format_exc()}


# ================================================================
if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
