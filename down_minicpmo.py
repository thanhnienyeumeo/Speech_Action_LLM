from transformers import AutoModel, BitsAndBytesConfig
import torch

model = AutoModel.from_pretrained(
    'openbmb/MiniCPM-o-2_6',
    trust_remote_code=True,
    attn_implementation='sdpa', # sdpa or flash_attention_2
    torch_dtype=torch.bfloat16,
    init_vision=False,
    init_audio=True,
    init_tts=True,
    # quantization_config=quant_config,
    low_cpu_mem_usage=True,
    device_map='cuda:0',
)