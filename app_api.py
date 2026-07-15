import os
import gc
import re
import requests
import base64
import numpy as np
import soundfile as sf
import torch
import torchaudio
torchaudio.set_audio_backend("soundfile")

# FastAPI for Local RunPod Mocking
from fastapi import FastAPI, Request, BackgroundTasks
import uvicorn
from voxcpm import VoxCPM

# 🛑 PyTorch Compile ပိတ်ခြင်း (Error ကာကွယ်ရန်)
import torch._dynamo
torch._dynamo.config.disable = True

# 📂 လမ်းကြောင်းများ သတ်မှတ်ခြင်း (ကိုကို့ Local လမ်းကြောင်းများ)
BASE_DIR = "/runpod-volume/VoxCPM2"
MODEL_DIR = os.path.join(BASE_DIR, "VoxCPM", "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 💡 Style Mode အတွက် အရန်ထားမည့် အသံဖိုင် (အရင် app.py က အသံဖိုင်)
GIRL_VOICE = os.path.join(BASE_DIR, "girl_voice.wav")
GIRL_PROMPT = "ချောမောတဲ့လူကတော့ တကယ်တော့ အကန့်အသတ်မရှိတဲ့ ဉာဏ်ရည်ဉာဏ်သွေးကို ပိုင်ဆိုင်ထားတဲ့ ထိပ်တန်းလိမ်လည်သူတစ်ယောက်ပဲ ဖြစ်ပါတယ်။ သူ့ရဲ့ အဓိကပစ်မှတ်ကတော့ ကိုရီးယားမှာ အကြီးမားဆုံး ငွေကြေးခဝါချမှုလုပ်ငန်းစုရဲ့ အကြီးအကဲတစ်ယောက်ပါပဲ။ ဒါပေမဲ့ လက်ရှိမှာတော့ အဲ့ဒီငွေကြေးခဝါချတဲ့သူဌေးက ထောင်ထဲရောက်နေပြီး အမြောက်အမြားရှိတဲ့ ငွေတွေဝှက်ထားတဲ့နေရာကတော့ လျှို့ဝှက်ချက်အဖြစ် ရှိနေဆဲဖြစ်ပါတယ်။"

# GPU Check
if torch.cuda.is_available():
    torch.set_default_device("cuda")
    print("🚀 NVIDIA GPU ဖြင့် အလုပ်လုပ်ပါမည်။")

print(f"⏳ Loading Model from {MODEL_DIR} ...")
model = VoxCPM.from_pretrained(MODEL_DIR, load_denoiser=False, local_files_only=True)
print("✅ Model loaded successfully!")

# ================================================================
# စာကြောင်းပိုင်းသည့်စနစ် (app.py အတိုင်း အတိအကျ သုံးထားသည်)
# ================================================================
def split_myanmar_text(text: str) -> list[str]:
    # ကွင်းစကွင်းပိတ်များကို ဖယ်ရှားမည်
    clean_text = re.sub(r'\[.*?\]', '', text)
    clean_text = re.sub(r'\(.*?\)', '', clean_text)
    # စာကြောင်းများကို ပိုင်းဖြတ်မည်
    smart_text = clean_text.replace('။', '။\n').replace('.', '.\n').replace('?', '?\n').replace('!', '!\n')
    target_texts = [t.strip() for t in smart_text.split('\n') if t.strip()]
    return target_texts

# ================================================================
# AI Generation Core (app.py အတိုင်း မူလအရည်အသွေးဖြင့် ထုတ်မည်)
# ================================================================
def generate_chunked(text: str, **kwargs) -> tuple[np.ndarray, int]:
    chunks = split_myanmar_text(text)
    
    # 💡 24000 အသေမထားဘဲ Model ၏ အမှန်တကယ် Sample Rate ကို ယူမည် (အသံမအောစေရန်)
    actual_sr = model.tts_model.sample_rate 
    silence_len = int(actual_sr * 0.5) 
    silence = np.zeros(silence_len, dtype=np.float32)
    audio_parts = []
    
    # 💡 app.py ကအတိုင်း အသေတ်မှတ်ချက်များ ထည့်မည် (၅၃၂ ခါ တွက်ခြင်းမှ ကာကွယ်ရန်)
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
                
            # 💡 float32 အဖြစ်သာ ထားမည် (အသံ မပီမသဖြစ်ခြင်းမှ ကာကွယ်ရန်)
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

# ================================================================
# API Handler Logic
# ================================================================
def handler(job_input):
    action    = job_input.get("action", "style")
    text      = job_input.get("text", "မင်္ဂလာပါ။")
    
    out_path  = os.path.join(OUTPUT_DIR, "output.wav")
    raw_ref   = os.path.join(OUTPUT_DIR, "raw_ref.wav")

    gen_kwargs = {}

    try:
        if action == "style":
            style = job_input.get("style", "")
            full_text = f"({style}){text}" if style else text
            
            # Style အတွက် Default Reference ကို app.py အတိုင်းထားမည်
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
            
            # Preprocess များကို ဖြုတ်ပြီး app.py အတိုင်း တိုက်ရိုက်သုံးမည်
            gen_kwargs["prompt_wav_path"] = raw_ref
            if reference_text: 
                gen_kwargs["prompt_text"] = reference_text

            final_wav, actual_sr = generate_chunked(text, **gen_kwargs)
            sf.write(out_path, final_wav, actual_sr)

        with open(out_path, "rb") as f:
            audio_base64 = base64.b64encode(f.read()).decode("utf-8")

        return {"status": "success", "audio_base64": audio_base64, "sample_rate": actual_sr}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# ================================================================
# FastAPI Server (Mocking RunPod)
# ================================================================
app = FastAPI()
jobs_db = {} # Job Status မှတ်သားရန်

@app.post("/run")
async def run_endpoint(background_tasks: BackgroundTasks, request: Request):
    data = await request.json()
    job_input = data.get("input", {})
    
    # Random Job ID ဖန်တီးမည်
    import uuid
    job_id = str(uuid.uuid4())
    jobs_db[job_id] = {"status": "IN_PROGRESS"}
    
    # Background တွင် AI ကို အလုပ်လုပ်ခိုင်းမည်
    def run_ai_task():
        try:
            result = handler(job_input)
            jobs_db[job_id] = {"status": "COMPLETED", "output": result}
        except Exception as e:
            jobs_db[job_id] = {"status": "FAILED", "error": str(e)}
            
    background_tasks.add_task(run_ai_task)
    return {"id": job_id, "status": "IN_QUEUE"}

@app.get("/status/{job_id}")
async def status_endpoint(job_id: str):
    return jobs_db.get(job_id, {"status": "FAILED", "error": "Job not found"})

if __name__ == "__main__":
    print("🌟 Local API Server စတင်နေပါပြီ... (http://0.0.0.0:8000)")
    uvicorn.run(app, host="0.0.0.0", port=8000)
