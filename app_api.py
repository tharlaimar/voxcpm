import os
import gc
import re
import time
import requests
import base64
import traceback
import numpy as np
import soundfile as sf
import torch
import torchaudio
torchaudio.set_audio_backend("soundfile")

import runpod
from voxcpm import VoxCPM

# 🛑 PyTorch Compile ပိတ်ခြင်း (Error ကာကွယ်ရန်)
import torch._dynamo
torch._dynamo.config.disable = True

# 📂 လမ်းကြောင်းများ သတ်မှတ်ခြင်း (ကိုကို့ Local လမ်းကြောင်းများ)
BASE_DIR = "/runpod-volume/VoxCPM2"
MODEL_DIR = os.path.join(BASE_DIR, "VoxCPM", "models")
OUTPUT_DIR = "/tmp"  # RunPod မှာ ယာယီဖိုင်တွေကို /tmp မှာ ထားတာ အကောင်းဆုံးပါ
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 💡 Style Mode အတွက် အရန်ထားမည့် အသံဖိုင်
GIRL_VOICE = os.path.join(BASE_DIR, "girl_voice.wav")
GIRL_PROMPT = "ချောမောတဲ့လူကတော့ တကယ်တော့ အကန့်အသတ်မရှိတဲ့ ဉာဏ်ရည်ဉာဏ်သွေးကို ပိုင်ဆိုင်ထားတဲ့ ထိပ်တန်းလိမ်လည်သူတစ်ယောက်ပဲ ဖြစ်ပါတယ်။ သူ့ရဲ့ အဓိကပစ်မှတ်ကတော့ ကိုရီးယားမှာ အကြီးမားဆုံး ငွေကြေးခဝါချမှုလုပ်ငန်းစုရဲ့ အကြီးအကဲတစ်ယောက်ပါပဲ။ ဒါပေမဲ့ လက်ရှိမှာတော့ အဲ့ဒီငွေကြေးခဝါချတဲ့သူဌေးက ထောင်ထဲရောက်နေပြီး အမြောက်အမြားရှိတဲ့ ငွေတွေဝှက်ထားတဲ့နေရာကတော့ လျှို့ဝှက်ချက်အဖြစ် ရှိနေဆဲဖြစ်ပါတယ်။"

# GPU Check
if torch.cuda.is_available():
    torch.set_default_device("cuda")
    print("🚀 NVIDIA GPU ဖြင့် အလုပ်လုပ်ပါမည်။")

print(f"⏳ Loading Model from {MODEL_DIR} ...")
try:
    model = VoxCPM.from_pretrained(MODEL_DIR, load_denoiser=False, local_files_only=True)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Model Loading Failed: {e}")
    model = None

# ================================================================
# စာကြောင်းပိုင်းသည့်စနစ်
# ================================================================
def split_myanmar_text(text: str) -> list[str]:
    clean_text = re.sub(r'\[.*?\]', '', text)
    clean_text = re.sub(r'\(.*?\)', '', clean_text)
    smart_text = clean_text.replace('။', '။\n').replace('.', '.\n').replace('?', '?\n').replace('!', '!\n')
    target_texts = [t.strip() for t in smart_text.split('\n') if t.strip()]
    return target_texts

# ================================================================
# AI Generation Core 
# ================================================================
def generate_chunked(text: str, **kwargs) -> tuple[np.ndarray, int]:
    chunks = split_myanmar_text(text)
    
    actual_sr = model.tts_model.sample_rate 
    silence_len = int(actual_sr * 0.5) 
    silence = np.zeros(silence_len, dtype=np.float32)
    audio_parts = []
    
    kwargs['cfg_value'] = 2.1
    kwargs['inference_timesteps'] = 15

    for i, chunk in enumerate(chunks):
        if len(chunk.strip()) < 2: continue
        
        with torch.inference_mode():
            safe_text = chunk + " "
            try:
                wav_chunk = model.generate(text=safe_text, **kwargs)
                
                if isinstance(wav_chunk, tuple):
                    wav_chunk = wav_chunk[0]
                if isinstance(wav_chunk, torch.Tensor):
                    wav_chunk = wav_chunk.detach().cpu().numpy()
                    
                wav_chunk = wav_chunk.astype(np.float32).flatten()
                audio_parts.append(wav_chunk)
                
                if i < len(chunks) - 1:
                    audio_parts.append(silence)
            except Exception as e:
                print(f"⚠️ Error generating chunk '{chunk}': {e}")
                # တစ်ပိုင်း Error တက်ရင်တောင် အကုန်မသေအောင် ကျော်သွားမယ်
                continue 
        
        # OOM မဖြစ်အောင် Memory ရှင်းထုတ်ခြင်း
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    if not audio_parts:
        return np.zeros(100, dtype=np.float32), actual_sr

    final_wav = np.concatenate(audio_parts)
    return final_wav, actual_sr

# ================================================================
# 💡 အသံဖိုင် Download ဆွဲရာတွင် (၃) ကြိမ်အထိ Retry လုပ်ပေးမည့် စနစ်
# ================================================================
def download_file(url: str, dest: str, max_retries: int = 3) -> None:
    for attempt in range(max_retries):
        try:
            print(f"📥 Downloading audio (Attempt {attempt + 1}/{max_retries})...")
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            with open(dest, "wb") as f:
                f.write(r.content)
            print("✅ Download successful!")
            return
        except Exception as e:
            print(f"⚠️ Download failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(2) # ၂ စက္ကန့်နားပြီးမှ ပြန်စမ်းမည်
            else:
                raise Exception(f"Failed to download audio after {max_retries} attempts. {e}")

# ================================================================
# API Handler Logic (RunPod)
# ================================================================
def handler(job):
    # Model ကြီး တက်မလာဘဲ Error တက်နေရင် ချက်ချင်း ပြန်ပို့ပေးမည်
    if model is None:
        return {"status": "error", "message": "WORKER_CRASHED: AI Model failed to load."}

    job_input = job.get("input", {})
    action    = job_input.get("action", "style")
    text      = job_input.get("text", "မင်္ဂလာပါ။")
    
    # 💡 ထပ်နေတဲ့ဖိုင်တွေ မဖြစ်အောင် Job ID လေးခံပြီး ယာယီနာမည်ပေးမည်
    job_id = job.get("id", "temp")
    out_path  = os.path.join(OUTPUT_DIR, f"output_{job_id}.wav")
    raw_ref   = os.path.join(OUTPUT_DIR, f"raw_ref_{job_id}.wav")

    gen_kwargs = {}

    try:
        if action == "style":
            style = job_input.get("style", "")
            full_text = f"({style}){text}" if style else text
            
            gen_kwargs["prompt_wav_path"] = GIRL_VOICE
            gen_kwargs["prompt_text"] = GIRL_PROMPT
            
            final_wav, actual_sr = generate_chunked(full_text, **gen_kwargs)
            sf.write(out_path, final_wav, actual_sr)

        elif action in ["preset", "clone"]:
            audio_url = job_input.get("audio_url")
            reference_text = job_input.get("reference_text", "").strip() 
            if not audio_url:
                raise Exception("audio_url is required")

            # 💡 (၃) ကြိမ် Retry ပါသော Download စနစ်ကို သုံးမည်
            download_file(audio_url, raw_ref)
            
            gen_kwargs["prompt_wav_path"] = raw_ref
            if reference_text: 
                gen_kwargs["prompt_text"] = reference_text

            final_wav, actual_sr = generate_chunked(text, **gen_kwargs)
            sf.write(out_path, final_wav, actual_sr)

        with open(out_path, "rb") as f:
            audio_base64 = base64.b64encode(f.read()).decode("utf-8")

        # 💡 ယာယီဖိုင်များကို ချက်ချင်း ပြန်ဖျက်မည် (Storage မပြည့်အောင်)
        try:
            if os.path.exists(out_path): os.remove(out_path)
            if os.path.exists(raw_ref): os.remove(raw_ref)
        except:
            pass

        return {"status": "success", "audio_base64": audio_base64, "sample_rate": actual_sr}
        
    except Exception as e:
        # Worker Crash မဖြစ်အောင် Traceback နဲ့ ဖမ်းထားပြီး Error အတိအကျကို ပြန်ပို့ပေးမည်
        err_msg = traceback.format_exc()
        print(f"❌ ERROR: {err_msg}")
        return {"status": "error", "message": str(e), "traceback": err_msg}

# ================================================================
# RunPod Serverless Entry Point
# ================================================================
if __name__ == "__main__":
    print("🌟 RunPod Serverless စတင်နေပါပြီ...")
    runpod.serverless.start({"handler": handler})
