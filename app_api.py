import runpod
import os
import torch
import base64
import torchaudio
from voxcpm import VoxCPMModel # VoxCPM library မှ တိုက်ရိုက်ခေါ်ခြင်း

# 1. ပတ်ဝန်းကျင်သတ်မှတ်ချက်
MODEL_PATH = "/runpod-volume/huggingface" # Network Volume ထဲက Model လမ်းကြောင်း

print("Loading VoxCPM Model...")
# Model ကို အစကတည်းက Load လုပ်ထားခြင်း
model = VoxCPMModel.from_pretrained(MODEL_PATH, device="cuda")
print("Model Loaded Successfully!")

def encode_audio_to_base64(audio_path):
    with open(audio_path, "rb") as audio_file:
        return base64.b64encode(audio_file.read()).decode('utf-8')

def handler(job):
    job_input = job['input']
    action = job_input.get("action")
    text = job_input.get("text", "မင်္ဂလာပါ")
    output_path = "/tmp/output.wav"

    try:
        # --- Mode 1: Preset Voice (အသင့်သုံးအသံ) ---
        if action == "preset":
            voice_id = job_input.get("voice_id", "default_speaker")
            # VoxCPM Preset Generation
            wav = model.generate(text=text, speaker_id=voice_id)
            torchaudio.save(output_path, wav.cpu(), 24000)

        # --- Mode 2: Voice Cloning (အသံကလုန်းခြင်း) ---
        elif action == "clone":
            # ကိုကို့ရဲ့ Clone Logic အရ reference_audio လိုအပ်ပါတယ်
            ref_audio = job_input.get("ref_audio_path", "/runpod-volume/voices/ref.wav")
            wav = model.clone(text=text, reference_audio=ref_audio)
            torchaudio.save(output_path, wav.cpu(), 24000)

        # --- Mode 3: Prompt/Style (Emotion ထည့်ထုတ်ခြင်း) ---
        elif action == "prompt":
            style = job_input.get("style", "neutral")
            wav = model.generate(text=text, style_prompt=style)
            torchaudio.save(output_path, wav.cpu(), 24000)

        else:
            return {"status": "error", "message": "Invalid action parameter"}

        # ရလဒ်ပြန်ပို့ခြင်း
        if os.path.exists(output_path):
            return {
                "status": "success",
                "audio_base64": encode_audio_to_base64(output_path)
            }
        else:
            return {"status": "error", "message": "Audio generation failed"}

    except Exception as e:
        return {"status": "error", "message": str(e)}

# RunPod Serverless စတင်ခြင်း
if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
