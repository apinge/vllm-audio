import torch
from transformers import pipeline

device = "cuda:0" if torch.cuda.is_available() else "cpu"


pipe = pipeline(
  "automatic-speech-recognition",
  model="/models/whisper-large-v3",
  chunk_length_s=30,
  device=device,
  torch_dtype=torch.float16, 
)

model = pipe.model  
model_dtype = model.dtype  
print(model_dtype)  # show torch.float16



transcription = pipe("Lisaan_first_clip_of_blessing.wav")['text']
print(transcription)
transcription = pipe("OSR_us_000_0010_8k.wav")['text']
print(transcription)
transcription = pipe("Allah_Wish.wav")['text']
print(transcription)
