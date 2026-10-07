from google.colab import drive
drive.mount('/content/drive')

!pip install voxcpm soundfile gradio -q

import torch
import torch._dynamo
torch._dynamo.config.disable = True

from voxcpm import VoxCPM
import gradio as gr
import soundfile as sf
import numpy as np
import gc
from datetime import datetime

print("⏳ Model ကို Drive ထဲကနေ မှတ်ဉာဏ်ထဲ ခေါ်နေပါပြီ...")
model = VoxCPM.from_pretrained("/content/drive/MyDrive/Channel99_Studio/VoxCPM/models", load_denoiser=False)
print("✅ Model ခေါ်လို့ ပြီးပါပြီ!")

# =========================================================================
# VOICE DATABASE (Preset)
# =========================================================================
VOICE_DATABASE = {
    
    "🗣️ ခိုင်ခိုင်": {
        "audio_path": "/content/drive/MyDrive/Channel99_Studio/ခိုင်ခိုင်.wav",
        "prompt_text": "ဒီနေ့ ပြောပြမယ့် အမှုကတော့၊ တကယ်ကို ထူးခြားဆန်းကြယ်ပြီး အဖြေရှာမရသေးတဲ့ အမှုတစ်ခုပဲ ဖြစ်ပါတယ်။"
    }
}

# =========================================================================
# CORE ENGINE (Preset + Clone နှစ်ခုစလုံး သုံးမယ်)
# =========================================================================
def run_tts(text_input, audio_path, prompt_text):
    gc.collect()
    torch.cuda.empty_cache()

    smart_text = text_input.replace('။', '။\n').replace('.', '.\n').replace('?', '?\n').replace('!', '!\n')
    target_texts = [t.strip() for t in smart_text.split('\n') if t.strip()]

    if not target_texts:
        return None

    all_wavs = []
    silence_len = int(model.tts_model.sample_rate * 0.15)
    silence = np.zeros(silence_len, dtype=np.float32)

    for i, text_chunk in enumerate(target_texts):
        if len(text_chunk) < 2:
            continue
        with torch.inference_mode():
            wav_chunk = model.generate(
                text=text_chunk + " ",
                prompt_wav_path=audio_path,
                prompt_text=prompt_text,
                cfg_value=2.1,
                inference_timesteps=15
            )
        all_wavs.append(wav_chunk)
        if i < len(target_texts) - 1:
            all_wavs.append(silence)
        torch.cuda.empty_cache()
        gc.collect()

    final_wav = np.concatenate(all_wavs)
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    return final_wav, model.tts_model.sample_rate, current_time

# =========================================================================
# TAB 1 — Preset Voice
# =========================================================================
def generate_preset(text_input, voice_choice):
    selected = VOICE_DATABASE.get(voice_choice, VOICE_DATABASE["🎙️  (Host)"])
    result = run_tts(text_input, selected["audio_path"], selected["prompt_text"])
    if result is None:
        return None

    final_wav, sr, ts = result
    clean_name = "".join([c for c in voice_choice if c.isalnum() or c.isspace()]).strip().replace(" ", "_")
    output_path = f"/content/drive/MyDrive/Channel99_Studio/outputs/preset_{clean_name}_{ts}.wav"
    sf.write(output_path, final_wav, sr)
    return output_path

# =========================================================================
# TAB 2 — Clone Voice (အသံဖိုင် + Reference Text ကိုယ်တိုင်ထည့်)
# =========================================================================
def generate_clone(text_input, uploaded_audio, reference_text):
    if uploaded_audio is None:
        return None, "⚠️ အသံဖိုင် တင်ပေးပါ။"
    if not reference_text.strip():
        return None, "⚠️ Reference Text ထည့်ပေးပါ။"

    result = run_tts(text_input, uploaded_audio, reference_text.strip())
    if result is None:
        return None, "⚠️ စာသား မပါပါ။"

    final_wav, sr, ts = result
    output_path = f"/content/drive/MyDrive/Channel99_Studio/outputs/clone_{ts}.wav"
    sf.write(output_path, final_wav, sr)
    return output_path, "✅ အသံထုတ်ပြီးပါပြီ!"

# =========================================================================
# TAB 3 — Prompt Voice
# =========================================================================
def generate_prompt_voice(text_input, style_prompt):
    if not style_prompt.strip():
        return None, "⚠️ Voice Style Prompt ထည့်ပေးပါ။"
    if not text_input.strip():
        return None, "⚠️ စာသား ထည့်ပေးပါ။"

    gc.collect()
    torch.cuda.empty_cache()

    # ── Style prompt ကို text ရှေ့မှာ () နဲ့ ထည့်လိုက်ရုံပဲ ──
    combined_text = f"({style_prompt.strip()}) {text_input.strip()}"

    smart_text = combined_text.replace('။', '။\n').replace('.', '.\n').replace('?', '?\n').replace('!', '!\n')
    target_texts = [t.strip() for t in smart_text.split('\n') if t.strip()]

    if not target_texts:
        return None, "⚠️ စာသား မပါပါ။"

    all_wavs = []
    silence_len = int(model.tts_model.sample_rate * 0.15)
    silence = np.zeros(silence_len, dtype=np.float32)

    for i, text_chunk in enumerate(target_texts):
        if len(text_chunk) < 2:
            continue
        with torch.inference_mode():
            wav_chunk = model.generate(
                text=text_chunk + " ",
                cfg_value=2.1,
                inference_timesteps=15
                # ── prompt_wav_path မလို၊ prompt_text မလို ──
            )
        all_wavs.append(wav_chunk)
        if i < len(target_texts) - 1:
            all_wavs.append(silence)
        torch.cuda.empty_cache()
        gc.collect()

    final_wav = np.concatenate(all_wavs)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"/content/drive/MyDrive/Channel99_Studio/outputs/prompt_{ts}.wav"
    sf.write(output_path, final_wav, model.tts_model.sample_rate)

    return output_path, "✅ အသံထုတ်ပြီးပါပြီ!"

# =========================================================================
# UI
# =========================================================================
with gr.Blocks(title="🎙️ Channel 99 AI Studio") as interface:
    gr.Markdown("# 🎙️ Channel 99 AI Studio")

    with gr.Tabs():

        # ── Tab 1: Preset ──
        with gr.Tab("🎭 Preset Voice"):
            with gr.Row():
                with gr.Column():
                    preset_text = gr.Textbox(
                        lines=10,
                        label="📝 စာသားထည့်ပါ",
                        placeholder="ဒီနေ့ ပြောပြမယ့် အမှုကတော့..."
                    )
                    preset_voice = gr.Radio(
                        choices=list(VOICE_DATABASE.keys()),
                        value="🎙️ ကိုကို (Host)",
                        label="🗣️ အသံရွေးပါ"
                    )
                    preset_btn = gr.Button("🎙️ အသံထုတ်မယ်", variant="primary")
                with gr.Column():
                    preset_output = gr.Audio(label="🎧 ထွက်လာမယ့် အသံ")

            preset_btn.click(
                fn=generate_preset,
                inputs=[preset_text, preset_voice],
                outputs=preset_output
            )

        # ── Tab 2: Clone ──
        with gr.Tab("🎤 Clone Voice"):
            with gr.Row():
                with gr.Column():
                    clone_text = gr.Textbox(
                        lines=10,
                        label="📝 စာသားထည့်ပါ",
                        placeholder="ဒီနေ့ ပြောပြမယ့် အမှုကတော့..."
                    )
                    clone_audio = gr.Audio(
                        label="🎵 Reference အသံဖိုင် တင်ပါ (WAV အကောင်းဆုံး)",
                        type="filepath"
                    )
                    clone_ref_text = gr.Textbox(
                        lines=4,
                        label="📄 Reference Text (အဲ့ဖိုင်ထဲမှာ ပြောထားတဲ့ စာသား)",
                        placeholder="Reference ဖိုင်ထဲမှာ ပြောထားတဲ့ စာသားကို အတိအကျ ထည့်ပါ..."
                    )
                    clone_btn = gr.Button("🎤 Clone ပြီး ထုတ်မယ်", variant="primary")
                with gr.Column():
                    clone_output = gr.Audio(label="🎧 ထွက်လာမယ့် အသံ")
                    clone_status = gr.Textbox(label="Status", interactive=False)

            clone_btn.click(
                fn=generate_clone,
                inputs=[clone_text, clone_audio, clone_ref_text],
                outputs=[clone_output, clone_status]
            )

            # ── Tab 3: Prompt Voice ──
        with gr.Tab("✨ Prompt Voice"):
            with gr.Row():
                with gr.Column():
                    prompt_text_input = gr.Textbox(
                        lines=10,
                        label="📝 စာသားထည့်ပါ",
                        placeholder="ဒီနေ့ ပြောပြမယ့် အမှုကတော့..."
                    )

                    gr.Markdown("### 💡 နမူနာ Prompt များ (နှိပ်လိုက်ရင် အလိုအလျောက် ထည့်ပေးမယ်)")

                    with gr.Row():
                        ex1 = gr.Button("🧑 Young man, calm & clear")
                        ex2 = gr.Button("👩 Young woman, warm & soft")
                    with gr.Row():
                        ex3 = gr.Button("📰 News anchor, formal & confident")
                        ex4 = gr.Button("📖 Storyteller, dramatic & slow")
                    with gr.Row():
                        ex5 = gr.Button("😄 Cheerful & energetic host")

                    prompt_style = gr.Textbox(
                        lines=3,
                        label="🎨 Voice Style Prompt",
                        placeholder="ဥပမာ - Young male voice, calm and clear, medium pace, deep tone"
                    )
                    prompt_btn = gr.Button("✨ Prompt နဲ့ အသံထုတ်မယ်", variant="primary")

                with gr.Column():
                    prompt_output = gr.Audio(label="🎧 ထွက်လာမယ့် အသံ")
                    prompt_status = gr.Textbox(label="Status", interactive=False)

            # နမူနာ Prompt များ Click လုပ်ရင် Textbox ထဲ ထည့်ပေး
            ex1.click(fn=lambda: "A young male voice, calm and clear, medium pace, deep tone", outputs=prompt_style)
            ex2.click(fn=lambda: "A young female voice, crystal clear, professional and articulate, confident tone", outputs=prompt_style)
            ex3.click(fn=lambda: "A news anchor voice, formal and authoritative, steady pace, perfect for broadcast", outputs=prompt_style)
            ex4.click(fn=lambda: "A storyteller voice, dramatic and expressive, slow pace, rich and warm tone", outputs=prompt_style)
            ex5.click(fn=lambda: "A cheerful host voice, energetic and friendly, fast pace, bright and engaging", outputs=prompt_style)

            prompt_btn.click(
                fn=generate_prompt_voice,
                inputs=[prompt_text_input, prompt_style],
                outputs=[prompt_output, prompt_status]
            )

print("🚀 Web UI စတင်နေပါပြီ...")
interface.launch(share=True, debug=True)
