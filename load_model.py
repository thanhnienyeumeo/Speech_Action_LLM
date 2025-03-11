from speech2text import SpeechToTextModel
import torch


device = torch.device('cuda')
path = 'speech_to_text_model_final'
model = SpeechToTextModel.from_pretrained(path)
model.eval()
model.to(device)
from datasets import load_dataset
import datasets
ds = load_dataset('Colder203/Audio_Robot_Interaction', split = 'train')

id = 0
audio_file = ds[id]['audio']['array']
sampling_rate = ds[id]['audio']['sampling_rate']
#or
# audio_file = 'audio/345.wav'
if isinstance(audio_file, str):
    import librosa
    audio_array, sampling_rate = librosa.load(audio_file, sr=16000)
    audio = {"array": audio_array, "sampling_rate": sampling_rate}
    
    # Process the audio
    audio_input = model.processor(
        audio=audio["array"], 
        sampling_rate=audio["sampling_rate"], 
        return_tensors="pt"
    ).input_features.to(device)
    
    # Generate text
    transcription = model.generate_from_audio(
        audio_input=audio_input,
        max_length=100,
        num_beams=4,
        temperature=0.7
    )
    
    print(f"Transcription: {transcription[0]}")
else:
    #using librosa to load an array as audio
    audio_array = audio_file
    audio = {"array": audio_array, "sampling_rate": sampling_rate}
    
    # Process the audio
    audio_input = model.processor(
        audio=audio["array"], 
        sampling_rate=audio["sampling_rate"], 
        return_tensors="pt"
    ).input_features.to(device)
    
    # Generate text
    transcription = model.generate_from_audio(
        audio_input=audio_input,
        max_length=300,
        num_beams=4,
        temperature=0.7
    )

    print(f"Transcription: {transcription[0]}")

