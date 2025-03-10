from speech2text import SpeechToTextModel
import torch


device = torch.device('cuda')
path = 'speech_to_text_model_final'
model = SpeechToTextModel.from_pretrained(path)
model.eval()
model.to(device)

audio_file = 'audio/345.wav'
if True:
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