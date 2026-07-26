import runpod
import os
import sys
import gc
import re
import requests
import base64
import numpy as np
import soundfile as sf
import traceback

import torch
import torchaudio

# 🛑 💡 VoxCPM က အတင်း Compile လုပ်နေတာကို လှည့်စားပြီး ပိတ်ပစ်မည်
def dummy_compile(model, *args, **kwargs):
    return model
torch.compile = dummy_compile

import torch._dynamo
torch._dynamo.config.disable = True

# 📂 လမ်းကြောင်းများ သတ်မှတ်ခြင်း
BASE_DIR = "/runpod-volume"
sys.path.append(BASE_DIR)
from voxcpm import VoxCPM 

MODEL_DIR = "/runpod-volume/VoxCPM2"
OUTPUT_DIR = "/tmp/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 💡 Style Mode အတွက် Google Cloud ကုဒ်အတိုင်း အရန်ထားမည့် အသံဖိုင်
GIRL_VOICE = os.path.join(BASE_DIR, "girl_voice.wav")
GIRL_PROMPT = "ချောမောတဲ့လူကတော့ တကယ်တော့ အကန့်အသတ်မရှိတဲ့ ဉာဏ်ရည်ဉာဏ်သွေးကို ပိုင်ဆိုင်ထားတဲ့ ထိပ်တန်းလိမ်လည်သူတစ်ယောက်ပဲ ဖြစ်ပါတယ်။ သူ့ရဲ့ အဓိကပစ်မှတ်ကတော့ ကိုရီးယားမှာ အကြီးမားဆုံး ငွေကြေးခဝါချမှုလုပ်ငန်းစုရဲ့ အကြီးအကဲတစ်ယောက်ပါပဲ။ ဒါပေမဲ့ လက်ရှိမှာတော့ အဲ့ဒီငွေကြေးခဝါချတဲ့သူဌေးက ထောင်ထဲရောက်နေပြီး အမြောက်အမြားရှိတဲ့ ငွေတွေဝှက်ထားတဲ့နေရာကတော့ လျှို့ဝှက်ချက်အဖြစ် ရှိနေဆဲဖြစ်ပါတယ်။"

if torch.cuda.is_available():
    torch.set_default_device("cuda")

model = None

def load_model_if_needed():
    global model
    if model is None:
        print(f"⏳ Loading Model from {MODEL_DIR} ...")
        model = VoxCPM.from_pretrained(MODEL_DIR, load_denoiser=False, local_files_only=True)
        print("✅ Model loaded successfully!")

def split_myanmar_text(text: str) -> list[str]:
    clean_text = re.sub(r'\[.*?\]', '', text)
    clean_text = re.sub(r'\(.*?\)', '', clean_text)
    smart_text = clean_text.replace('။', '။\n').replace('.', '.\n').replace('?', '?\n').replace('!', '!\n')
    target_texts = [t.strip() for t in smart_text.split('\n') if t.strip()]
    return target_texts

def generate_chunked(text: str, **kwargs) -> tuple[np.ndarray, int]:
    load_model_if_needed()
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
            wav_chunk = model.generate(text=safe_text, **kwargs)
            
            if isinstance(wav_chunk, tuple):
                wav_chunk = wav_chunk[0]
            if isinstance(wav_chunk, torch.Tensor):
                wav_chunk = wav_chunk.detach().cpu().numpy()
                
            wav_chunk = wav_chunk.astype(np.float32).flatten()
            audio_parts.append(wav_chunk)
            
            if i < len(chunks) - 1:
                audio_parts.append(silence)
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    if not audio_parts:
        return np.zeros(100, dtype=np.float32), actual_sr

    final_wav = np.concatenate(audio_parts)
    return final_wav, actual_sr

def download_file(url: str, dest: str) -> None:
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    with open(dest, "wb") as f:
        f.write(r.content)

def handler(job):
    job_input = job.get("input", {})
    action    = job_input.get("action", "style")
    text      = job_input.get("text", "မင်္ဂလာပါ။")
    
    out_path  = os.path.join(OUTPUT_DIR, "output.wav")
    raw_ref   = os.path.join(OUTPUT_DIR, "raw_ref.wav")

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

            download_file(audio_url, raw_ref)
            
            gen_kwargs["prompt_wav_path"] = raw_ref
            if reference_text: 
                gen_kwargs["prompt_text"] = reference_text

            final_wav, actual_sr = generate_chunked(text, **gen_kwargs)
            sf.write(out_path, final_wav, actual_sr)

        with open(out_path, "rb") as f:
            audio_base64 = base64.b64encode(f.read()).decode("utf-8")

        return {"status": "success", "audio_base64": audio_base64, "sample_rate": actual_sr}
        
    except Exception as e:
        return {"status": "error", "message": str(e), "traceback": traceback.format_exc()}

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
