import torch
import torchaudio
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from datasets import load_dataset


device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "/models/whisper-large-v3"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype,
)
model.to(device)
model.eval()

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
)


waveform, sr = torchaudio.load("out.wav")
if waveform.shape[0] > 1:
    waveform = torch.mean(waveform, dim=0, keepdim=True)  # 转成 [1, time]


target_sr = pipe.feature_extractor.sampling_rate
if sr != target_sr:
    waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)(waveform)

batch_size = 32
batched_waveforms = [waveform.numpy()] * batch_size  # [ ndarray, ndarray, ... ]

#batched_waveforms =  waveform.expand(batch_size, -1)
#result = pipe(batched_waveforms, batch_size=batch_size)
import time
start = time.time()
result = pipe(batched_waveforms)
duration = time.time() - start 

print(f"len(prompts):{len(batched_waveforms)}")
print("Duration:", duration)
print("RPS:", len(batched_waveforms) / duration)

assert len(result) == batch_size
print(result)

