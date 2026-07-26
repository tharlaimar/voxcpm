import runpod
import os
import sys
import base64
import numpy as np
import soundfile as sf
import traceback
import torch

# ================================================================
# Model Loading (Patch တွေ လုံးဝမလိုတော့ပါ)
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
# Handler (Colab ကဲ့သို့ အရှင်းလင်းဆုံးစနစ်)
# ================================================================
def handler(job):
    job_input = job.get("input", {})
    
    style = job_input.get("style", "A young female voice with a crystal clear, highly professional tone.")
    text = job_input.get("text", "ဒီနေ့ ပြောပြမယ့် အမှုကတော့၊ တကယ်ကို ထူးခြားဆန်းကြယ်ပြီး အဖြေရှာမရသေးတဲ့ အမှုတစ်ခုပဲ ဖြစ်ပါတယ်။")
    
    out_path = os.path.join(OUTPUT_DIR, "output.wav")

    try:
        load_model_if_needed()
        
        # 💡 Colab လိုပဲ (Style) Text ပုံစံဖြင့် တိုက်ရိုက်တွဲမည်
        text_to_speak = f"({style.strip()}) {text.strip()}"
        
        with torch.inference_mode():
            # 💡 prompt_wav_path တွေ, Normalize တွေ လုံးဝမလိုတော့ပါ
            wav_chunk = model.generate(
                text=text_to_speak,
                cfg_value=2.0,
                inference_timesteps=15
            )
        
        # Tensor မှ Numpy သို့ ပြောင်းခြင်း
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
