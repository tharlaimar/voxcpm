import numpy as np
import soundfile as sf
import torch
import torchaudio
torchaudio.set_audio_backend("soundfile") 
from voxcpm import VoxCPM

# GPU ရဲ့ Tensor Core များကို အပြည့်အဝ အသုံးပြုရန် (Speed Up)
torch.set_float32_matmul_precision('high')
# 🚨 [တရားခံ ဖယ်ရှားခြင်း] torch.set_float32_matmul_precision('high') ကို ဖြုတ်ပစ်လိုက်ပါပြီ (ဂြိုဟ်သားသံ မထွက်စေရန်)

# 🌟 [FAIL-FAST SYSTEM အဆင့်မြှင့်တင်ခြင်း]
# sys.exit(1) ဖြင့် အတင်းမပိတ်ဘဲ၊ Flutter ဆီ ချက်ချင်း Error ပို့ရန် Flag မှတ်ထားပါမည်
gpu_is_bad = not torch.cuda.is_available()
model = None
MODEL_PATH  = "/runpod-volume/VoxCPM2"


if gpu_is_bad:
    print("[CRITICAL] CUDA/GPU မမိပါ! (Driver အဟောင်းဖြစ်နေနိုင်သည်)")
    print("[CRITICAL] Worker သည် ဝင်လာသမျှ Job များကို ချက်ချင်း Error ပြန်ပို့ပါမည်။")
    print("[CRITICAL] CUDA/GPU မမိပါ!")
else:
    # GPU ကောင်းမှသာ Model ကို Load လုပ်ပါမည် (CPU ပေါ် Load မိပြီး Freeze ဖြစ်ခြင်းမှ ကာကွယ်ရန်)
    print(f"[INIT] Loading VoxCPM2 from {MODEL_PATH} ...")
    # 💡 Colab အတိုင်း load_denoiser=False ဖြင့်သာ ပြန်ခေါ်ပါမည်
    model = VoxCPM.from_pretrained(MODEL_PATH, load_denoiser=False)
    print("[INIT] Model loaded successfully!")

# ================================================================
# Text Chunking နှင့် Generate Function များကို (အရင်အတိုင်း ဆက်ထားပါ)
# Audio Preprocessor
# ================================================================
# def split_myanmar_text(...):
# def generate_chunked(...):
# def encode_audio(...):
# def download_file(...):

# ================================================================
# Handler
# ================================================================
def handler(job):
    # 🚀 ဝင်လာတာနဲ့ GPU မကောင်းရင် Flutter ဆီ ချက်ချင်း Error ပြန်ကန်ထုတ်မည်
    if gpu_is_bad:
        return {"status": "error", "message": "WORKER_CRASHED"}

    job_input = job["input"]
    action    = job_input.get("action", "style")
    text      = job_input.get("text", "မင်္ဂလာပါ။")
    out_path  = "/tmp/output.wav"
def preprocess_audio(input_path: str, output_path: str) -> str:
    try:
        import torchaudio.transforms as T
        wav, sr = torchaudio.load(input_path)
        if wav.shape[0] > 1:
            wav = torch.mean(wav, dim=0, keepdim=True)
        if sr != 24000:
            resampler = T.Resample(orig_freq=sr, new_freq=24000)
            wav = resampler(wav)
        torchaudio.save(output_path, wav, 24000)
        return output_path
    except Exception as e:
        print(f"[AUDIO ERROR] {e}")
        return input_path

    # ... (ကျန်တဲ့ ကုဒ်တွေ အကုန် အရင်အတိုင်းပါပဲ ကိုကို) ...
# ================================================================
# Text Chunking (မြန်မာ/အင်္ဂလိပ်)
# Text Chunking
# ================================================================
def split_myanmar_text(text: str, max_chars: int = MAX_CHARS) -> list[str]:
    """
    မြန်မာစာကို (။၊) နဲ့ အင်္ဂလိပ်စာကို (.,) နဲ့ ဖြတ်ပြီး chunk တွေ ခွဲတယ်။
    chunk တစ်ခု max_chars ထက် မကျော်ဘဲ တတ်နိုင်သမျှ ပေါင်းသိမ်းတယ်။
    """
    # 💡 အင်္ဂလိပ်စာလုံးတွေ အလယ်ကနေ အတင်းမဖြတ်မိအောင် Full stop (.) နဲ့ Comma (,) ပါ ထည့်ထားပါတယ်
    sentences = re.split(r'(?<=[။၊.,])', text)
    sentences = [s.strip() for s in sentences if s.strip()]

    chunks = []
    current = ""

    for sentence in sentences:
        if len(sentence) > max_chars:
            if current:
@@ -72,53 +59,48 @@ def split_myanmar_text(text: str, max_chars: int = MAX_CHARS) -> list[str]:
            for i in range(0, len(sentence), max_chars):
                chunks.append(sentence[i:i+max_chars].strip())
            continue

        if len(current) + len(sentence) <= max_chars:
            current += sentence
        else:
            if current:
                chunks.append(current.strip())
            current = sentence

    if current:
        chunks.append(current.strip())

    return [c for c in chunks if c]


# ================================================================
# Generation System
# ================================================================
def generate_chunked(text: str, **kwargs) -> tuple[np.ndarray, int]:
    """
    Long text ကို chunk တွေ ခွဲပြီး generate လုပ်ပြီး concatenate လုပ်တယ်။
    """
    chunks = split_myanmar_text(text)
    print(f"[CHUNK] Split into {len(chunks)} chunks: {[len(c) for c in chunks]} chars")

    # 💡 VoxCPM ရဲ့ မှန်ကန်တဲ့ Sample Rate ကို ယူပါမယ်
    
    try:
        actual_sr = int(model.tts_model.sample_rate)
    except:
        actual_sr = 24000 # အရန်အနေနဲ့ ၂၄၀၀၀ ထားပါမယ်
        actual_sr = 24000 

    silence_len = int(actual_sr * 0.15)
    silence = np.zeros(silence_len, dtype=np.float32)

    audio_parts = []

    for i, chunk in enumerate(chunks):
        if len(chunk.strip()) < 2:
            continue

        print(f"[CHUNK] Generating chunk {i+1}/{len(chunks)}...")

        with torch.inference_mode():
            safe_text = chunk + " "
            wav = model.generate(text=safe_text, **kwargs)
            raw_output = model.generate(text=safe_text, **kwargs)

            # Tuple ပြန်လာလျှင် အသံဖိုင်ကိုသာ တိတိကျကျ ယူပါမည်
            if isinstance(raw_output, tuple):
                wav = raw_output[0]
            else:
                wav = raw_output
                
            if isinstance(wav, torch.Tensor):
                wav = wav.detach().cpu().numpy()
                
            wav = np.array(wav).astype(np.float32).flatten()

            audio_parts.append(wav)
@@ -130,126 +112,95 @@ def generate_chunked(text: str, **kwargs) -> tuple[np.ndarray, int]:
    if not audio_parts:
        return np.zeros(100, dtype=np.int16), actual_sr

    # အပိုင်းတွေကို ပေါင်းပါမယ်
    combined = np.concatenate(audio_parts)

    # 🚀 [အရေးကြီးဆုံး ပြင်ဆင်ချက်: Audio Normalization] 🚀
    # အသံလှိုင်း (Waveform) က Volume အရမ်းကျယ်နေပြီး Clipping ဖြစ်နေတာကို 
    # ပုံမှန်အသံဖြစ်အောင် အချိုးကျ ပြန်ချုံ့ပေးပါမယ် (Normalize)
    
    # 💡 [အသံ Volume ညှိခြင်း]
    max_val = np.max(np.abs(combined))
    if max_val > 0:
        combined = combined / max_val  # အသံကို -1.0 နှင့် 1.0 ကြားရောက်အောင် ညှိပါမည်
        combined = combined / max_val  

    # Flutter က အကောင်းဆုံး နားလည်တဲ့ Standard 16-bit PCM (WAV) format သို့ ပြောင်းပါမည်
    combined = (combined * 32767).astype(np.int16)

    return combined, actual_sr
    # 🚀 [အရေးကြီးဆုံး ပြင်ဆင်ချက်] Flutter အတွက် Standard 16-bit PCM အဖြစ် တိတိကျကျ ပြောင်းပေးခြင်း
    combined_int16 = (combined * 32767).astype(np.int16)
    
    return combined_int16, actual_sr

# ================================================================
# Helper functions
# ================================================================
def encode_audio(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def download_file(url: str, dest: str) -> None:
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    with open(dest, "wb") as f:
        f.write(r.content)


# ================================================================
# Handler
# ================================================================
def handler(job):
    if gpu_is_bad:
        return {"status": "error", "message": "WORKER_CRASHED"}

    job_input = job["input"]
    action    = job_input.get("action", "style")
    text      = job_input.get("text", "မင်္ဂလာပါ။")
    out_path  = "/tmp/output.wav"

    # 💡 Colab အတိုင်း timesteps ကို 15 ပြန်ထားပါမည်
    gen_kwargs = dict(
        cfg_value=2.0,
        inference_timesteps=10,
        inference_timesteps=15, 
    )

    try:

        # ── Mode 1: Style ─────────────────────────────────────────
        if action == "style":
            style     = job_input.get("style", "")
            style = job_input.get("style", "")
            full_text = f"({style}){text}" if style else text
            wav_int16, actual_sr = generate_chunked(full_text, **gen_kwargs)
            # int16 ဖြစ်သွားသည့်အတွက် subtype ထည့်စရာမလိုတော့ဘဲ တိုက်ရိုက် Save ပါမည်
            sf.write(out_path, wav_int16, actual_sr)

            wav, actual_sr = generate_chunked(full_text, **gen_kwargs)
            sf.write(out_path, wav, actual_sr)

        # ── Mode 2: Preset ────────────────────────────────────────
        elif action == "preset":
        elif action in ["preset", "clone"]:
            audio_url = job_input.get("audio_url")
            if not audio_url:
                return {"status": "error", "message": "audio_url is required for preset action"}
            reference_text = job_input.get("reference_text", "").strip() 

            ref_path = "/tmp/preset_ref.wav"
            download_file(audio_url, ref_path)
            if action == "preset" and not reference_text:
                reference_text = "ဒီနေ့ ပြောပြမယ့် အမှုကတော့၊ တကယ်ကို ထူးခြားဆန်းကြယ်ပြီး အဖြေရှာမရသေးတဲ့ အမှုတစ်ခုပဲ ဖြစ်ပါတယ်။"

            # 💡 prompt_text မရှိလျှင် prompt_wav_path မထည့်ပါ (library error ရှောင်ရန်)
            wav, actual_sr = generate_chunked(
                text,
                prompt_wav_path=ref_path,
                    prompt_text="ဒီနေ့ ပြောပြမယ့် အမှုကတော့၊ တကယ်ကို ထူးခြားဆန်းကြယ်ပြီး အဖြေရှာမရသေးတဲ့ အမှုတစ်ခုပဲ ဖြစ်ပါတယ်။",
                **gen_kwargs,
            )
            sf.write(out_path, wav, actual_sr)
            if not audio_url:
                return {"status": "error", "message": "audio_url is required"}

        # ── Mode 3: Clone ─────────────────────────────────────────
        elif action == "clone":
            audio_url      = job_input.get("audio_url")
            reference_text = job_input.get("reference_text", "")
            raw_ref = "/tmp/raw_ref.wav"
            clean_ref = "/tmp/clean_ref.wav"
            download_file(audio_url, raw_ref)
            
            prompt_wav_path = preprocess_audio(raw_ref, clean_ref)

            if not audio_url:
                return {"status": "error", "message": "audio_url is required for clone action"}

            ref_path = "/tmp/clone_ref.wav"
            download_file(audio_url, ref_path)

            if reference_text:
                # 💡 နှစ်ခုလုံးရှိလျှင် တွဲထည့်ပါမည်
                wav, actual_sr = generate_chunked(
                    text,
                    prompt_wav_path=ref_path,
                    prompt_text=reference_text,
                    **gen_kwargs,
                )
            else:
                # 💡 prompt_text မရှိလျှင် ဘာမှမထည့်ပါ
                wav, actual_sr = generate_chunked(
                    text,
                    **gen_kwargs,
                )
            sf.write(out_path, wav, actual_sr)
            kwargs = dict(gen_kwargs)
            kwargs["prompt_wav_path"] = prompt_wav_path
            
            if reference_text: 
                kwargs["prompt_text"] = reference_text

            wav_int16, actual_sr = generate_chunked(text, **kwargs)
            sf.write(out_path, wav_int16, actual_sr)

        else:
            return {
                "status":  "error",
                "message": f"Unknown action '{action}'. Use 'style', 'preset', or 'clone'."
            }
            return {"status": "error", "message": f"Unknown action '{action}'"}

        # ── Response ──────────────────────────────────────────────
        if os.path.exists(out_path):
            return {
                "status":       "success",
                "status": "success",
                "audio_base64": encode_audio(out_path),
                "sample_rate":  actual_sr, # 💡 မှန်ကန်သော sample rate ကို ပြန်ပို့ပါမည်
                "sample_rate": actual_sr,
            }
        else:
            return {"status": "error", "message": "Audio file was not created"}
            return {"status": "error", "message": "Audio file not created"}

    except Exception as e:
        import traceback
        return {"status": "error", "message": str(e), "traceback": traceback.format_exc()}

        err_msg = traceback.format_exc()
        return {"status": "error", "message": str(e), "traceback": err_msg}

# ================================================================
if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
