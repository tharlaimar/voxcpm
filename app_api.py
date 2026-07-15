import runpod
import os
import re
import requests
import base64
import gc
import traceback
import numpy as np
import soundfile as sf
import torch
import torchaudio
torchaudio.set_audio_backend("soundfile")

from voxcpm import VoxCPM

# ================================================================
# PyTorch Compile ပိတ်ခြင်း
# ================================================================
import torch._dynamo
torch._dynamo.config.disable = True

# ================================================================
# Path များ
# ================================================================
MODEL_PATH  = "/runpod-volume/VoxCPM2"


# Style mode အတွက် default reference (runpod-volume ထဲ ထည့်ထားရမယ်)
GIRL_VOICE  = "/runpod-volume/girl_voice.wav"
GIRL_PROMPT = (
    "ချောမောတဲ့လူကတော့ တကယ်တော့ အကန့်အသတ်မရှိတဲ့ ဉာဏ်ရည်ဉာဏ်သွေးကို "
    "ပိုင်ဆိုင်ထားတဲ့ ထိပ်တန်းလိမ်လည်သူတစ်ယောက်ပဲ ဖြစ်ပါတယ်။ သူ့ရဲ့ "
    "အဓိကပစ်မှတ်ကတော့ ကိုရီးယားမှာ အကြီးမားဆုံး ငွေကြေးခဝါချမှုလုပ်ငန်းစုရဲ့ "
    "အကြီးအကဲတစ်ယောက်ပါပဲ။ ဒါပေမဲ့ လက်ရှိမှာတော့ အဲ့ဒီငွေကြေးခဝါချတဲ့သူဌေးက "
    "ထောင်ထဲရောက်နေပြီး အမြောက်အမြားရှိတဲ့ ငွေတွေဝှက်ထားတဲ့နေရာကတော့ "
    "လျှို့ဝှက်ချက်အဖြစ် ရှိနေဆဲဖြစ်ပါတယ်။"
)

# ================================================================
# GPU Check
# ================================================================
if torch.cuda.is_available():
    torch.set_default_device("cuda")
    print("🚀 NVIDIA GPU ဖြင့် အလုပ်လုပ်ပါမည်။")
else:
    print("⚠️ GPU မတွေ့ပါ — CPU ဖြင့် run မည်။")

# ================================================================
# Model Load — တစ်ကြိမ်တည်းပဲ load မည်
# ================================================================
print(f"[INIT] Loading VoxCPM2 from {MODEL_PATH} ...")
try:
    model = VoxCPM.from_pretrained(MODEL_PATH, load_denoiser=False)
    print("[INIT] ✅ Model loaded successfully!")
except Exception as e:
    print(f"[INIT] ❌ Model load failed: {e}")
    model = None

# ================================================================
# မြန်မာစာ Chunking
# ================================================================
def split_myanmar_text(text: str, max_chars: int = MAX_CHARS) -> list:
    sentences = re.split(r'(?<=[။၊])', text)
    sentences = [s.strip() for s in sentences if s.strip()]

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
# Chunked Generation
# ================================================================
def generate_chunked(text: str, **kwargs) -> tuple:
    chunks = split_myanmar_text(text)
    print(f"[CHUNK] {len(chunks)} chunks: {[len(c) for c in chunks]}")

    # local_api.py နဲ့ same — model sample rate အတိုင်းယူမည်
    actual_sr = model.tts_model.sample_rate
    silence_len = int(actual_sr * 0.5)
    silence = np.zeros(silence_len, dtype=np.float32)

    kwargs['cfg_value'] = 2.1
    kwargs['inference_timesteps'] = 15

    audio_parts = []

    for i, chunk in enumerate(chunks):
        if len(chunk.strip()) < 2:
            continue

        print(f"[CHUNK] Generating {i+1}/{len(chunks)}: {chunk[:40]}...")

        with torch.inference_mode():
            try:
                wav_chunk = model.generate(text=chunk + " ", **kwargs)

                if isinstance(wav_chunk, tuple):
                    wav_chunk = wav_chunk[0]
                if isinstance(wav_chunk, torch.Tensor):
                    wav_chunk = wav_chunk.detach().cpu().numpy()

                wav_chunk = wav_chunk.astype(np.float32).flatten()
                audio_parts.append(wav_chunk)

                if i < len(chunks) - 1:
                    audio_parts.append(silence)

            except Exception as e:
                print(f"⚠️ Chunk {i+1} error: {e} — ကျော်သွားမည်")
                continue

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    if not audio_parts:
        return np.zeros(100, dtype=np.float32), actual_sr

    return np.concatenate(audio_parts), actual_sr


# ================================================================
# Helper
# ================================================================
def download_file(url: str, dest: str, max_retries: int = 3) -> None:
    for attempt in range(max_retries):
        try:
            print(f"📥 Downloading (attempt {attempt+1}/{max_retries}): {url}")
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            with open(dest, "wb") as f:
                f.write(r.content)
            print("✅ Download OK")
            return
        except Exception as e:
            print(f"⚠️ Download failed: {e}")
            if attempt < max_retries - 1:
                import time
                time.sleep(2)
            else:
                raise Exception(f"Download failed after {max_retries} attempts: {e}")


def encode_audio(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# ================================================================
# Handler
# ================================================================
def handler(job):
    if model is None:
        return {"status": "error", "message": "Model failed to load — worker crashed"}

    job_input = job.get("input", {})
    action    = job_input.get("action", "style")
    text      = job_input.get("text", "မင်္ဂလာပါ။")
    job_id    = job.get("id", "temp")

    out_path = f"/tmp/output_{job_id}.wav"
    ref_path = f"/tmp/ref_{job_id}.wav"

    gen_kwargs = {}

    try:
        # ── Mode 1: Style ─────────────────────────────────────
        if action == "style":
            style     = job_input.get("style", "")
            full_text = f"({style}){text}" if style else text

            gen_kwargs["prompt_wav_path"] = GIRL_VOICE
            gen_kwargs["prompt_text"]     = GIRL_PROMPT

            final_wav, actual_sr = generate_chunked(full_text, **gen_kwargs)
            sf.write(out_path, final_wav, actual_sr)

        # ── Mode 2: Preset ────────────────────────────────────
        elif action == "preset":
            audio_url = job_input.get("audio_url")
            if not audio_url:
                return {"status": "error", "message": "audio_url is required"}

            download_file(audio_url, ref_path)

            gen_kwargs["prompt_wav_path"] = ref_path

            final_wav, actual_sr = generate_chunked(text, **gen_kwargs)
            sf.write(out_path, final_wav, actual_sr)

        # ── Mode 3: Clone ─────────────────────────────────────
        elif action == "clone":
            audio_url      = job_input.get("audio_url")
            reference_text = job_input.get("reference_text", "").strip()

            if not audio_url:
                return {"status": "error", "message": "audio_url is required"}

            download_file(audio_url, ref_path)

            gen_kwargs["prompt_wav_path"] = ref_path
            if reference_text:
                gen_kwargs["prompt_text"] = reference_text

            final_wav, actual_sr = generate_chunked(text, **gen_kwargs)
            sf.write(out_path, final_wav, actual_sr)

        else:
            return {"status": "error", "message": f"Unknown action: {action}"}

        # ── Response ──────────────────────────────────────────
        audio_b64 = encode_audio(out_path)

        # temp files ဖျက်မည်
        for p in [out_path, ref_path]:
            try:
                if os.path.exists(p):
                    os.remove(p)
            except:
                pass

        return {
            "status":       "success",
            "audio_base64": audio_b64,
            "sample_rate":  actual_sr,
        }

    except Exception as e:
        err = traceback.format_exc()
        print(f"❌ ERROR:\n{err}")
        return {"status": "error", "message": str(e), "traceback": err}


# ================================================================
if __name__ == "__main__":
    print("🌟 RunPod Serverless starting...")
    runpod.serverless.start({"handler": handler})
