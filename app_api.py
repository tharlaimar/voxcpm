# ၀။ Google Drive ကို အရင်ဆုံး အလိုအလျောက် ချိတ်ပါမယ်
from google.colab import drive
drive.mount('/content/drive')

# ၁။ Library တွေ သွင်းမယ်
!pip install -q "funasr==1.4.12" "voxcpm==2.0.3" "soundfile" "gradio==6.20.0"

import torch
# 🛑 Error ကို ဖြစ်စေတဲ့ PyTorch ရဲ့ Compile စနစ်ကြီးကို အမြစ်ပြတ် ပိတ်မယ်
import torch._dynamo
torch._dynamo.config.disable = True

from voxcpm import VoxCPM
import gradio as gr
import soundfile as sf
import numpy as np
import gc
from datetime import datetime

print("⏳ Model ကို Drive ထဲကနေ မှတ်ဉာဏ်ထဲ ခေါ်နေပါပြီ (ခဏလေး စောင့်ပေးပါ)...")
# ၂။ Model ခေါ်မယ်
model = VoxCPM.from_pretrained("/content/drive/MyDrive/Channel99_Studio/VoxCPM/models", load_denoiser=False)
print("✅ Model ခေါ်လို့ ပြီးပါပြီ!")


# =========================================================================
# ⚙️ 💡 [ကိုကို ပြင်ရမယ့်နေရာ] - အသံအသစ် ၉ မျိုးရဲ့ အချက်အလက်များကို ဒီမှာ စာရင်းသွင်းပါ
# =========================================================================
# အသံဖိုင်လမ်းကြောင်း (audio_path) နဲ့ အဲ့ဒီအသံဖိုင်ထဲက ပြောထားတဲ့စာသား (prompt_text) ကို တွဲပေးထားရပါမယ်။
VOICE_DATABASE = {
    
    "🗣️ ခိုင်ခိုင်": {
        "audio_path": "/content/drive/MyDrive/Channel99_Studio/ခိုင်ခိုင်.wav",
        "prompt_text": "ဒီနေ့ ပြောပြမယ့် အမှုကတော့၊ တကယ်ကို ထူးခြားဆန်းကြယ်ပြီး အဖြေရှာမရသေးတဲ့ အမှုတစ်ခုပဲ ဖြစ်ပါတယ်။"
    },
}
# =========================================================================


# ၃။ အသံထုတ်မယ့် အင်ဂျင်
def generate_channel99_voice(text_input, voice_choice):
    gc.collect()
    torch.cuda.empty_cache()

    # 💡 ရွေးချယ်လိုက်တဲ့ အသံအလိုက် ဒေတာတွေကို Database ထဲကနေ အလိုအလျောက် ဆွဲထုတ်ပါမယ်
    selected_voice = VOICE_DATABASE.get(voice_choice)

    if not selected_voice:
        print("⚠️ ရွေးချယ်ထားသော အသံကို မတွေ့ရှိပါ။ Default အသံဖြင့် မောင်းပါမည်။")
        selected_voice = VOICE_DATABASE["🎙️ ကိုကို (Host)"]

    prompt_audio_path = selected_voice["audio_path"]
    prompt_text = selected_voice["prompt_text"]

    # ဉာဏ်ကောင်းတဲ့ စာကြောင်းပိုင်းဖြတ်စနစ်
    smart_text = text_input.replace('။', '။\n').replace('.', '.\n').replace('?', '?\n').replace('!', '!\n')
    target_texts = [t.strip() for t in smart_text.split('\n') if t.strip()]

    if not target_texts:
        return None

    all_wavs = []

    # ကြားထဲမှာ ဟဟကြီး မဖြစ်အောင် 0.5 ကနေ 0.15 စက္ကန့်ကို လျှော့လိုက်ပါပြီ
    silence_len = int(model.tts_model.sample_rate * 0.15)
    silence = np.zeros(silence_len, dtype=np.float32)

    for i, text_chunk in enumerate(target_texts):
        if len(text_chunk) < 2:
            continue

        with torch.inference_mode():
            # စာလုံး အမြီးမပြတ်အောင် နောက်ဆုံးမှာ Space တစ်ချက် အလိုလို ခံပေးမယ့်စနစ်
            safe_text = text_chunk + " "

            wav_chunk = model.generate(
                text=safe_text,
                prompt_wav_path=prompt_audio_path,
                prompt_text=prompt_text,
                cfg_value=2.1,
                inference_timesteps=15
            )

        all_wavs.append(wav_chunk)
        if i < len(target_texts) - 1:
            all_wavs.append(silence)

        # KV Cache ပြည့်တဲ့ Error မတက်အောင် Loop တစ်ခါပတ်တိုင်း Memory ရှင်းပေးပါမယ်
        torch.cuda.empty_cache()
        gc.collect()

    final_wav = np.concatenate(all_wavs)

    # အချိန်နဲ့ ဖိုင်နာမည် မှတ်မယ် (ဖိုင်နာမည် ရှုပ်မသွားအောင် ရွေးတဲ့အသံရဲ့ နာမည်ကို သန့်စင်ပြီး ထည့်ပါမယ်)
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    clean_speaker_name = "".join([c for c in voice_choice if c.isalnum() or c.isspace()]).strip().replace(" ", "_")
    output_path = f"/content/drive/MyDrive/Channel99_Studio/outputs/channel99_{clean_speaker_name}_{current_time}.wav"

    sf.write(output_path, final_wav, model.tts_model.sample_rate)

    return output_path

# ၄။ Web UI ဒီဇိုင်း
print("🚀 Web UI ကို စတင် ဖွင့်နေပါပြီ...")
interface = gr.Interface(
    fn=generate_channel99_voice,
    inputs=[
        gr.Textbox(
            lines=10,
            label="📝 ဒီမှာ စာသားတွေ ထည့်ပါ ကိုကို",
            placeholder="ဥပမာ - \nဒီနေ့ ပြောပြမယ့် အမှုကတော့... \nတကယ်ကို ထူးခြားဆန်းကြယ်တဲ့ အမှုတစ်ခုပါ။"
        ),
        gr.Radio(
            # 💡 UI ရဲ့ Choices ကို Database ရဲ့ Keys တွေကနေ အလိုအလျောက် ဆွဲယူပြပေးမှာဖြစ်လို့ စာရင်းအကုန် ပါဝင်နေမှာပါ
            choices=list(VOICE_DATABASE.keys()),
            value="🎙️ ကိုကို (Host)",
            label="🗣️ ဘယ်သူ့အသံနဲ့ ထုတ်မလဲ ရွေးပါ ကိုကို"
        )
    ],
    outputs=gr.Audio(label="🎧 ထွက်လာမယ့် အသံ (ဒီကနေ တန်းနားထောင်လို့ရပါပြီ)"),
    title="🎙️ Channel 99 AI Studio (Master Version)",
    description="အသံမြန်တာ၊ စာသားပြတ်တာ နဲ့ Memory ပြည့်တာတွေကို အပြည့်အဝ ဖြေရှင်းထားတဲ့ အကောင်းဆုံး ဗားရှင်းပါ။"
)

# ၅။ UI ကို လွှင့်မယ်
interface.launch(share=True, debug=True)
