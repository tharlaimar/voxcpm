import runpod
import os
import requests
import base64
import soundfile as sf
from voxcpm import VoxCPM

# ================================================================
# Model ကို Global scope မှာ load လုပ်တယ်
# Worker start တဲ့အချိန် တစ်ကြိမ်ပဲ run မယ် — request တိုင်း မဟုတ်ဘူး
# ================================================================
MODEL_PATH = "/runpod-volume/VoxCPM2"

print(f"[INIT] Loading VoxCPM2 from {MODEL_PATH} ...")
model = VoxCPM.from_pretrained(MODEL_PATH, load_denoiser=False)
print("[INIT] Model loaded successfully!")


# ================================================================
# Helper functions
# ================================================================
def encode_audio_to_base64(audio_path: str) -> str:
    with open(audio_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def download_audio(url: str, dest_path: str) -> None:
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    with open(dest_path, "wb") as f:
        f.write(response.content)


# ================================================================
# Main handler — RunPod က request တိုင်း ဒီ function ကို call မယ်
#
# Input format:
#
# Mode 1 — Voice Design (reference audio မလို):
# {
#   "input": {
#     "action": "design",
#     "text": "မင်္ဂလာပါ ကျွန်တော် VoxCPM ကို သုံးနေပါတယ်",
#     "style": "A warm female voice, gentle and calm"   ← optional
#   }
# }
#
# Mode 2 — Voice Clone (reference audio URL လို):
# {
#   "input": {
#     "action": "clone",
#     "text": "မင်္ဂလာပါ ကျွန်တော် VoxCPM ကို သုံးနေပါတယ်",
#     "audio_url": "https://example.com/reference_voice.wav"
#   }
# }
#
# Output:
# {
#   "status": "success",
#   "audio_base64": "...",   ← base64 encoded WAV
#   "sample_rate": 48000
# }
# ================================================================
def handler(job):
    job_input   = job["input"]
    action      = job_input.get("action", "design")
    text        = job_input.get("text", "Hello from VoxCPM2!")
    output_path = "/tmp/output.wav"

    try:
        # ── Mode 1: Voice Design ──────────────────────────────────
        if action == "design":
            style     = job_input.get("style", "")
            full_text = f"({style}){text}" if style else text

            wav = model.generate(
                text=full_text,
                cfg_value=2.0,
                inference_timesteps=10,
            )
            sf.write(output_path, wav, model.tts_model.sample_rate)

        # ── Mode 2: Voice Clone ───────────────────────────────────
        elif action == "clone":
            audio_url = job_input.get("audio_url")
            if not audio_url:
                return {"status": "error", "message": "audio_url is required for clone action"}

            ref_path = "/tmp/reference.wav"
            download_audio(audio_url, ref_path)

            wav = model.generate(
                text=text,
                reference_wav_path=ref_path,
                cfg_value=2.0,
                inference_timesteps=10,
            )
            sf.write(output_path, wav, model.tts_model.sample_rate)

        else:
            return {"status": "error", "message": f"Unknown action '{action}'. Use 'design' or 'clone'."}

        # ── Response ──────────────────────────────────────────────
        if os.path.exists(output_path):
            return {
                "status":       "success",
                "audio_base64": encode_audio_to_base64(output_path),
                "sample_rate":  model.tts_model.sample_rate,
            }
        else:
            return {"status": "error", "message": "Audio file was not created"}

    except Exception as e:
        return {"status": "error", "message": str(e)}


# ================================================================
if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
