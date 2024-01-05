import gradio as gr
import torch, torchaudio
import os
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
import numpy as np

xtts_model_path = "/root/models/coqui/XTTS-v2/"
config = XttsConfig()
config.load_json(os.path.join(xtts_model_path, "config.json"))
model = Xtts.init_from_config(config)
model.load_checkpoint(config, checkpoint_dir=xtts_model_path, eval=True)
model.cuda()

sample_rate = 24000
output_wav_path = '/root/tutorials/output/output.wav'

def predict(message):
    text = message
    outputs = model.synthesize(
        text,
        config,
        speaker_wav=os.path.join(xtts_model_path, "samples", "en_sample.wav"),
        gpt_cond_len=3,
        language="zh-cn",
    )
    torchaudio.save(output_wav_path, torch.tensor(outputs["wav"]).unsqueeze(0), sample_rate)
    return output_wav_path

if __name__ == '__main__':
    iface = gr.Interface(fn=predict, inputs="text", outputs="audio", title="Text To Voice")
    iface.launch(server_name='0.0.0.0')

