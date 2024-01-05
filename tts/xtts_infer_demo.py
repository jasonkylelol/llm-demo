import os
import torch, torchaudio
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

model_path = '/root/models/coqui/XTTS-v2/'
#speaker_wav_path = '/root/models/coqui/XTTS-v2/samples/en_sample.wav'
speaker_wav_path = '/root/models/coqui/XTTS-v2/samples/zh-cn-sample.wav'

# generate speech by cloning a voice using default settings
#text = "It took me quite a long time to develop a voice, and now that I have it I'm not going to be silent."
text = '一只敏捷的棕色狐狸跳过了一只懒狗'

output_wav_path = '/root/tutorials/output/output.wav'

config = XttsConfig()
config.load_json(os.path.join(model_path, 'config.json'))

model = Xtts.init_from_config(config)
model.load_checkpoint(config, checkpoint_dir=model_path, eval=True)
model.cuda()

outputs = model.synthesize(
    text,
    config,
    speaker_wav=[speaker_wav_path],
    gpt_cond_len=3,
    language="zh",
)

torchaudio.save(output_wav_path, torch.tensor(outputs["wav"]).unsqueeze(0), 24000)

