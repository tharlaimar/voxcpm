import runpod
import os
import sys
import base64
import numpy as np
import soundfile as sf
import traceback

import torch
import torch.nn.functional as F
import torchaudio

# ================================================================
# 🛑 BUG FIX: Audio Corruption / Static Noise (Float16 Overflow ပြဿနာဖြေရှင်းခြင်း)
# ================================================================
def safe_torchaudio_load(filepath, *args, **kwargs):
    data, sr = sf.read(filepath)
    if data.ndim > 1: data = data.mean(axis=1)
    tensor = torch.from_numpy(data).float().unsqueeze(0)
    return tensor, sr
torchaudio.load = safe_torchaudio_load

_original_sdpa = F.scaled_dot_product_attention

def safe_sdpa(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False, **kwargs):
    # 1. ညှိပေးခြင်း (Shape Matching)
    if query.size(1) != key.size(1):
        num_groups = query.size(1) // key.size(1)
        key = key.repeat_interleave(num_groups, dim=1)
        value = value.repeat_interleave(num_groups, dim=1)
        
    try:
        sdpa_kwargs = {"attn_mask": attn_mask, "dropout_p": dropout_p, "is_causal": is_causal}
        if scale is not None: sdpa_kwargs["scale"] = scale
        return _original_sdpa(query, key, value, **sdpa_kwargs)
    except Exception:
        # 💡 အဓိက ပြင်ဆင်ချက်: Overflow မဖြစ်စေရန် (ခွေးအူသံ ပျောက်စေရန်) Math တွက်ချက်မှုများကို Float32 ဖြင့်သာ ပြုလုပ်မည်
        q = query.to(torch.float32)
        k = key.to(torch.float32)
        v = value.to(torch.float32)
        
        scale_factor = scale if scale is not None else (1.0 / (q.size(-1) ** 0.5))
        attn_weight = torch.matmul(q, k.transpose(-2, -1)) * scale_factor
        
        if is_causal:
            L, S = q.size(-2), k.size(-2)
            causal_mask = torch.ones((L, S), dtype=torch.bool, device=q.device).tril()
            attn_weight = attn_weight.masked_fill(~causal_mask, -torch.inf)
            
        if attn_mask is not None: 
            attn_weight = attn_weight + attn_mask.to(torch.float32)
            
        attn_weight = torch.softmax(attn_weight, dim=-1)
        out = torch.matmul(attn_weight, v)
        
        return out.to(query.dtype) # မူလ Type သို့ ပြန်ပြောင်းမည်

F.scaled_dot_product_attention = safe_sdpa

def dummy_compile(model, *args, **kwargs): return model
torch.compile = dummy_compile
import torch._dynamo
torch._dynamo.config.disable = True

# ================================================================
# Model Loading
# ================================================================
BASE_DIR = "/runpod-volume"
sys.path.append(BASE_DIR)
from voxcpm import VoxCPM 

MODEL_DIR = "/runpod-volume/VoxCPM2"
OUTPUT_DIR = "/tmp/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

model = None

def load_model_if_needed():
    global model
    if model is None:
        print("⏳ Loading Model...")
        model = VoxCPM.from_pretrained(MODEL_DIR, load_denoiser=False, local_files_only=True)
        if torch.cuda.is_available():
            for attr_name, attr_value in vars(model).items():
                if isinstance(attr_value, torch.nn.Module):
                    attr_value.to("cuda")
                elif isinstance(attr_value, torch.Tensor):
                    setattr(model, attr_name, attr_value.to("cuda"))
            if hasattr(model, 'device'): model.device = "cuda"
            if hasattr(model, 'tts_model') and hasattr(model.tts_model, 'device'):
                model.tts_model.device = "cuda"

# ================================================================
# Handler (Colab လို ရိုးရှင်းသော စနစ်)
# ================================================================
def handler(job):
    job_input = job.get("input", {})
    
    style = job_input.get("style", "A young female voice with a crystal clear, highly professional tone.")
    text = job_input.get("text", "ဒီနေ့ ပြောပြမယ့် အမှုကတော့၊ တကယ်ကို ထူးခြားဆန်းကြယ်ပြီး အဖြေရှာမရသေးတဲ့ အမှုတစ်ခုပဲ ဖြစ်ပါတယ်။")
    
    out_path = os.path.join(OUTPUT_DIR, "output.wav")

    try:
        load_model_if_needed()
        text_to_speak = f"({style.strip()}) {text.strip()}"
        
        with torch.inference_mode():
            wav_chunk = model.generate(
                text=text_to_speak,
                cfg_value=2.0,
                inference_timesteps=15
            )
        
        if isinstance(wav_chunk, tuple):
            wav_chunk = wav_chunk[0]
        if hasattr(wav_chunk, 'detach'):
            wav_chunk = wav_chunk.detach().cpu().numpy()
            
        wav_chunk = np.array(wav_chunk, dtype=np.float32).flatten()
        
        sf.write(out_path, wav_chunk, model.tts_model.sample_rate)

        with open(out_path, "rb") as f:
            audio_base64 = base64.b64encode(f.read()).decode("utf-8")

        return {"status": "success", "audio_base64": audio_base64, "sample_rate": model.tts_model.sample_rate}
        
    except Exception as e:
        return {"status": "error", "message": str(e), "traceback": traceback.format_exc()}

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
