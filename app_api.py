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
import torch.nn.functional as F
import torchaudio

# ================================================================
# 🛑 💡 MAGIC PATCH 1: TorchCodec Bypass (Soundfile ဖြင့် torchaudio.load ကို အစားထိုးခြင်း)
# ================================================================
def safe_torchaudio_load(filepath, *args, **kwargs):
    data, sr = sf.read(filepath)
    if data.ndim == 1:
        tensor = torch.from_numpy(data).float().unsqueeze(0)
    else:
        tensor = torch.from_numpy(data.T).float()
    return tensor, sr

torchaudio.load = safe_torchaudio_load

# ================================================================
# 🛑 💡 MAGIC PATCH 2: PyTorch Attention Bug အပြီးတိုင်ဖြေရှင်းခြင်း (GQA ပါ)
# ================================================================
_original_sdpa = F.scaled_dot_product_attention

def safe_sdpa(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False, **kwargs):
    if query.size(1) != key.size(1):
        num_groups = query.size(1) // key.size(1)
        key = key.repeat_interleave(num_groups, dim=1)
        value = value.repeat_interleave(num_groups, dim=1)
    
    try:
        sdpa_kwargs = {
            "attn_mask": attn_mask,
            "dropout_p": dropout_p,
            "is_causal": is_causal,
        }
        if scale is not None:
            sdpa_kwargs["scale"] = scale
            
        return _original_sdpa(query, key, value, **sdpa_kwargs)
    except Exception:
        scale_factor = scale if scale is not None else (1.0 / (query.size(-1) ** 0.5))
        attn_weight = torch.matmul(query, key.transpose(-2, -1)) * scale_factor
        
        if is_causal:
            L, S = query.size(-2), key.size(-2)
            causal_mask = torch.ones((L, S), dtype=torch.bool, device=query.device).tril()
            attn_weight = attn_weight.masked_fill(~causal_mask, -10000.0)
        
        if attn_mask is not None:
            attn_weight = attn_weight + attn_mask
            
        attn_weight = torch.softmax(attn_weight, dim=-1)
        return torch.matmul(attn_weight, value)

F.scaled_dot_product_attention = safe_sdpa

def dummy_compile(model, *args, **kwargs):
    return model
torch.compile = dummy_compile

import torch._dynamo
torch._dynamo.config.disable = True
# ================================================================

BASE_DIR = "/runpod-volume"
sys.path.append(BASE_DIR)
from voxcpm import VoxCPM 

MODEL_DIR = "/runpod-volume/VoxCPM2"
OUTPUT_DIR = "/tmp/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

GIRL_VOICE = os.path.join(BASE_DIR, "koko_voice.wav")
GIRL_PROMPT = "ချောမောတဲ့လူကတော့ တကယ်တော့ အကန့်အသတ်မရှိတဲ့ ဉာဏ်ရည်ဉာဏ်သွေးကို ပိုင်ဆိုင်ထားတဲ့ ထိပ်တန်းလိမ်လည်သူတစ်ယောက်ပဲ ဖြစ်ပါတယ်။ သူ့ရဲ့ အဓိကပစ်မှတ်ကတော့ ကိုရီးယားမှာ အကြီးမားဆုံး ငွေကြေးခဝါချမှုလုပ်ငန်းစုရဲ့ အကြီးအကဲတစ်ယောက်ပါပဲ။ ဒါပေမဲ့ လက်ရှိမှာတော့ အဲ့ဒီငွေကြေးခဝါချတဲ့သူဌေးက ထောင်ထဲရောက်နေပြီး အမြောက်အမြားရှိတဲ့ ငွေတွေဝှက်ထားတဲ့နေရာကတော့ လျှို့ဝှက်ချက်အဖြစ် ရှိနေဆဲဖြစ်ပါတယ်။"

model = None

def load_model_if_needed():
    global model
    if model is None:
        print(f"⏳ Loading Model from {MODEL_DIR} ...")
        
        model = VoxCPM.from_pretrained(MODEL_DIR, load_denoiser=False, local_files_only=True)
            
        if torch.cuda.is_available():
            for attr_name, attr_value in vars(model).items():
                if isinstance(attr_value, torch.nn.Module):
                    attr_value.to("cuda")
                elif isinstance(attr_value, torch.Tensor):
                    setattr(model, attr_name, attr_value.to("cuda"))
                    
            if hasattr(model, 'device'):
                model.device = "cuda"
            if hasattr(model, 'tts_model') and hasattr(model.tts_model, 'device'):
                model.tts_model.device = "cuda"
                
            print("🚀 Model's internal modules safely moved to NVIDIA GPU!")
            
        print("✅ Model loaded successfully!")

# 📂 စာကြောင်းခွဲသည့်စနစ်များ (Prompt Voice အတွက် သီးသန့်နှင့် ပုံမှန်အတွက် သီးသန့်)
def split_myanmar_text_with_parentheses(text: str) -> list[str]:
    clean_text = re.sub(r'\[.*?\]', '', text)
    smart_text = clean_text.replace('။', '။\n').replace('.', '.\n').replace('?', '?\n').replace('!', '!\n')
    target_texts = [t.strip() for t in smart_text.split('\n') if t.strip()]
    return target_texts

def split_myanmar_text(text: str) -> list[str]:
    clean_text = re.sub(r'\[.*?\]', '', text)
    clean_text = re.sub(r'\(.*?\)', '', clean_text)
    smart_text = clean_text.replace('။', '။\n').replace('.', '.\n').replace('?', '?\n').replace('!', '!\n')
    target_texts = [t.strip() for t in smart_text.split('\n') if t.strip()]
    return target_texts

def generate_chunked_prompt_voice(combined_text: str) -> tuple[np.ndarray, int]:
    load_model_if_needed()
    chunks = split_myanmar_text_with_parentheses(combined_text)
    
    actual_sr = model.tts_model.sample_rate 
    silence_len = int(actual_sr * 0.15) 
    silence = np.zeros(silence_len, dtype=np.float32)
    audio_parts = []
    
    for i, chunk in enumerate(chunks):
        if len(chunk.strip()) < 2: continue
        
        with torch.inference_mode():
            safe_text = chunk + " "
            wav_chunk = model.generate(
                text=safe_text,
                cfg_value=2.1,
                inference_timesteps=15
            )
            
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
        if action in ["style", "prompt_style"]:
            style_value = job_input.get("style") or job_input.get("style_prompt", "")
            
            if style_value and not style_value.endswith(".wav"):
                combined_text = f"({style_value.strip()}) {text.strip()}"
                final_wav, actual_sr = generate_chunked_prompt_voice(combined_text)
            else:
                full_text = f"({style_value}){text}" if style_value else text
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
        error_trace = traceback.format_exc()
        print("CRITICAL ERROR TRACEBACK:\n", error_trace)
        return {
            "status": "error", 
            "message": str(e), 
            "traceback": error_trace
        }

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
