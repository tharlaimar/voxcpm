import runpod
import os
import requests
import base64
import soundfile as sf
from voxcpm import VoxCPM

# ====== Network Volume Path ======
# မောင်ရဲ့ volume ထဲ model ဘယ် folder မှာ ထည့်ထားလဲ?
MODEL_PATH = "/runpod-volume/VoxCPM2"

print(f"Loading VoxCPM2 from {MODEL_PATH}...")
model = VoxCPM.from_pretrained(MODEL_PATH, load_denoiser=False)
print("Model loaded successfully!")

# ==========================================

def encode_audio_to_base64(audio_path):
    with open(audio_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def handler(job):
    job_input = job["input"]
    action    = job_input.get("action", "design")
    text      = job_input.get("text", "မင်္ဂလာပါ")
    output_path = "/tmp/output.wav"

    try:
        # --- Mode 1: Voice Design (preset style) ---
        if action == "design":
            style = job_input.get("style", "")
            full_text = f"({style}){text}" if style else text
            wav = model.generate(
                text=full_text,
                cfg_value=2.0,
                inference_timesteps=10,
            )
            sf.write(output_path, wav, model.tts_model.sample_rate)

        # --- Mode 2: Voice Clone ---
        elif action == "clone":
            audio_url = job_input.get("audio_url")
            ref_path  = "/tmp/reference.wav"
            r = requests.get(audio_url, timeout=30)
            with open(ref_path, "wb") as f:
                f.write(r.content)
            wav = model.generate(
                text=text,
                reference_wav_path=ref_path,
                cfg_value=2.0,
                inference_timesteps=10,
            )
            sf.write(output_path, wav, model.tts_model.sample_rate)

        else:
            return {"status": "error", "message": f"Unknown action: {action}"}

        if os.path.exists(output_path):
            return {
                "status": "success",
                "audio_base64": encode_audio_to_base64(output_path),
                "sample_rate": model.tts_model.sample_rate,
            }
        else:
            return {"status": "error", "message": "Audio file not created"}

    except Exception as e:
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
