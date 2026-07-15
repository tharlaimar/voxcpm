import runpod
import os
import re
import requests
import base64
import numpy as np
import soundfile as sf
import torch
import torchaudio
torchaudio.set_audio_backend("soundfile") 
from voxcpm import VoxCPM

# 🚨 [တရားခံ ဖယ်ရှားခြင်း] torch.set_float32_matmul_precision('high') ကို ဖြုတ်ပစ်လိုက်ပါပြီ (ဂြိုဟ်သားသံ မထွက်စေရန်)

gpu_is_bad = not torch.cuda.is_available()
model = None
MODEL_PATH  = "/runpod-volume/VoxCPM2"
MAX_CHARS   = 150

if gpu_is_bad:
    print("[CRITICAL] CUDA/GPU မမိပါ!")
else:
    print(f"[INIT] Loading VoxCPM2 from {MODEL_PATH} ...")
    # 💡 Colab အတိုင်း load_denoiser=False ဖြင့်သာ ပြန်ခေါ်ပါမည်
    model = VoxCPM.from_pretrained(MODEL_PATH, load_denoiser=False)
    print("[INIT] Model loaded successfully!")

# ================================================================
# Audio Preprocessor
# ================================================================
def preprocess_audio(input_path: str, output_path: str) -> str:
    try:
        import torchaudio.transforms as T
        wav, sr = torchaudio.load(input_path)
        if wav.shape[0] > 1:
            wav = torch.mean(wav, dim=0, keepdim=True)
        if sr != 24000:
            resampler = T.Resample(orig_freq=sr, new_freq=24000)
            wav = resampler(wav)
        torchaudio.save(output_path, wav, 24000)
        return output_path
    except Exception as e:
        print(f"[AUDIO ERROR] {e}")
        return input_path

# ================================================================
# Text Chunking
# ================================================================
def split_myanmar_text(text: str, max_chars: int = MAX_CHARS) -> list[str]:
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
    chunks = split_myanmar_text(text)
    
    try:
        actual_sr = int(model.tts_model.sample_rate)
    except:
        actual_sr = 24000 
        
    silence_len = int(actual_sr * 0.15)
    silence = np.zeros(silence_len, dtype=np.float32)
    audio_parts = []
    
    for i, chunk in enumerate(chunks):
        if len(chunk.strip()) < 2:
            continue
            
        with torch.inference_mode():
            safe_text = chunk + " "
            raw_output = model.generate(text=safe_text, **kwargs)
            
            # Tuple ပြန်လာလျှင် အသံဖိုင်ကိုသာ တိတိကျကျ ယူပါမည်
            if isinstance(raw_output, tuple):
                wav = raw_output[0]
            else:
                wav = raw_output
                
            if isinstance(wav, torch.Tensor):
                wav = wav.detach().cpu().numpy()
                
            wav = np.array(wav).astype(np.float32).flatten()
            
            audio_parts.append(wav)
            if i < len(chunks) - 1:
                audio_parts.append(silence)
                
        torch.cuda.empty_cache()

    if not audio_parts:
        return np.zeros(100, dtype=np.int16), actual_sr

    combined = np.concatenate(audio_parts)
    
    # 💡 [အသံ Volume ညှိခြင်း]
    max_val = np.max(np.abs(combined))
    if max_val > 0:
        combined = combined / max_val  
        
    # 🚀 [အရေးကြီးဆုံး ပြင်ဆင်ချက်] Flutter အတွက် Standard 16-bit PCM အဖြစ် တိတိကျကျ ပြောင်းပေးခြင်း
    combined_int16 = (combined * 32767).astype(np.int16)
    
    return combined_int16, actual_sr

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
    if gpu_is_bad:
        return {"status": "error", "message": "WORKER_CRASHED"}

    job_input = job["input"]
    action    = job_input.get("action", "style")
    text      = job_input.get("text", "မင်္ဂလာပါ။")
    out_path  = "/tmp/output.wav"

    # 💡 Colab အတိုင်း timesteps ကို 15 ပြန်ထားပါမည်
    gen_kwargs = dict(
        cfg_value=2.0,
        inference_timesteps=15, 
    )

    try:
        if action == "style":
            style = job_input.get("style", "")
            full_text = f"({style}){text}" if style else text
            wav_int16, actual_sr = generate_chunked(full_text, **gen_kwargs)
            # int16 ဖြစ်သွားသည့်အတွက် subtype ထည့်စရာမလိုတော့ဘဲ တိုက်ရိုက် Save ပါမည်
            sf.write(out_path, wav_int16, actual_sr)

        elif action in ["preset", "clone"]:
            audio_url = job_input.get("audio_url")
            reference_text = job_input.get("reference_text", "").strip() 

            if action == "preset" and not reference_text:
                reference_text = "ဒီနေ့ ပြောပြမယ့် အမှုကတော့၊ တကယ်ကို ထူးခြားဆန်းကြယ်ပြီး အဖြေရှာမရသေးတဲ့ အမှုတစ်ခုပဲ ဖြစ်ပါတယ်။"

            if not audio_url:
                return {"status": "error", "message": "audio_url is required"}

            raw_ref = "/tmp/raw_ref.wav"
            clean_ref = "/tmp/clean_ref.wav"
            download_file(audio_url, raw_ref)
            
            prompt_wav_path = preprocess_audio(raw_ref, clean_ref)

            kwargs = dict(gen_kwargs)
            kwargs["prompt_wav_path"] = prompt_wav_path
            
            if reference_text: 
                kwargs["prompt_text"] = reference_text

            wav_int16, actual_sr = generate_chunked(text, **kwargs)
            sf.write(out_path, wav_int16, actual_sr)

        else:
            return {"status": "error", "message": f"Unknown action '{action}'"}

        if os.path.exists(out_path):
            return {
                "status": "success",
                "audio_base64": encode_audio(out_path),
                "sample_rate": actual_sr,
            }
        else:
            return {"status": "error", "message": "Audio file not created"}

    except Exception as e:
        import traceback
        err_msg = traceback.format_exc()
        return {"status": "error", "message": str(e), "traceback": err_msg}

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
