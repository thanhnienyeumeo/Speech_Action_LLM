from transformers import AutoModel, BitsAndBytesConfig
import math
import numpy as np

from moviepy.editor import VideoFileClip
import tempfile
import librosa
import soundfile as sf
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer
from decord import VideoReader, cpu  

#download "openbmb/MiniCPM-o-2_6-int4"
model = AutoModel.from_pretrained(
    'openbmb/MiniCPM-o-2_6',
    trust_remote_code=True,
    attn_implementation='sdpa', # sdpa or flash_attention_2
    torch_dtype=torch.bfloat16,
    # init_vision=True,
    init_vision=False,
    init_audio=True,
    init_tts=True,
    # quantization_config=quant_config,
    low_cpu_mem_usage=True,
    device_map='cuda:0',
)

# model = model.eval().cuda()
tokenizer = AutoTokenizer.from_pretrained('openbmb/MiniCPM-o-2_6', trust_remote_code=True)

model.init_tts()
model.tts.float()
for file_id in range(300, 360):
    
    mimick_prompt = "Please repeat each user's speech, including voice style and speech content."
    audio_input, _ = librosa.load(f'audio/{file_id}.wav', sr=16000, mono=True) # load the audio to be mimicked

    # `./assets/input_examples/fast-pace.wav`, 
    # `./assets/input_examples/chi-english-1.wav` 
    # `./assets/input_examples/exciting-emotion.wav` 
    # for different aspects of speech-centric features.

    msgs = [{'role': 'user', 'content': [mimick_prompt, audio_input]}]
    res = model.chat(
        msgs=msgs,
        tokenizer=tokenizer,
        sampling=True,
        max_new_tokens=128,
        use_tts_template=True,
        temperature=0.8,
        generate_audio=True,
        output_audio_path=f'audio/mimick/{file_id}.wav', # save the tts result to output_audio_path
    )
    print(f'gen audio {file_id} done')

print('DONE')

