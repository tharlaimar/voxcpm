import runpod
import os
import base64
import torchaudio
from voxcpm import VoxCPMModel

# ၁။ Model Load လုပ်ခြင်း
print("Initializing Model...")
# Model Path ကို ကိုကို့ရဲ့ Network Volume လမ်းကြောင်းအတိုင်းထားပါ
model = VoxCPMModel.from_pretrained("/runpod-volume/huggingface", device="cuda")
print("Model Ready!")

# Base64 ပြောင်းပေးသည့် Function
def encode_audio_to_base64(audio_path):
    with open(audio_path, "rb") as audio_file:
        return base64.b64encode(audio_file.read()).decode('utf-8')

# ၂။ Handler Function (Request ဝင်လာတိုင်း အလုပ်လုပ်မည့်နေရာ)
def handler(job):
    job_input = job['input']
    action = job_input.get("action")
    text = job_input.get("text", "မင်္ဂလာပါ")
    output_path = "/tmp/output.wav"

    try:
        # Mode 1: Preset Voice
        if action == "preset":
            voice_id = job_input.get("voice_id", "default_speaker")
            wav = model.generate(text=text, speaker_id=voice_id)
            torchaudio.save(output_path, wav.cpu(), 24000)

        # Mode 2: Voice Cloning
        elif action == "clone":
            ref_audio = job_input.get("ref_audio_path")
            wav = model.clone(text=text, reference_audio=ref_audio)
            torchaudio.save(output_path, wav.cpu(), 24000)

        # Mode 3: Prompt/Style
        elif action == "prompt":
            style = job_input.get("style", "neutral")
            wav = model.generate(text=text, style_prompt=style)
            torchaudio.save(output_path, wav.cpu(), 24000)

        else:
            return {"status": "error", "message": "Invalid action"}

        return {"status": "success", "audio_base64": encode_audio_to_base64(output_path)}

    except Exception as e:
        return {"status": "error", "message": str(e)}

# ၃။ RunPod ကို ခေါ်ခြင်း (ဒီစာကြောင်းက အရေးကြီးဆုံးပါ)
if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
