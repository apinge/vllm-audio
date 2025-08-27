from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import time
from vllm import LLM, SamplingParams
from transformers import WhisperTokenizerFast
from pathlib import Path
from librosa import resample, load
import torch
import numpy as np
import sys
sys.stdout.reconfigure(encoding='utf-8')

def chunking(audio:np.ndarray, sample_rate:int):
    """Split audio to 30 second duration chunks

    Args:
        audio (np.ndarray): Audio data
        sample_rate (int): Audio sample rate
    """
    max_duration_samples = sample_rate * 30.0
    padding = max_duration_samples - np.remainder(len(audio), max_duration_samples)
    audio = np.pad(audio, (0, padding.astype(int)), 'constant', constant_values=0.0)
    return np.split(audio, len(audio) // max_duration_samples)



whisper = LLM(
        model="/app/xisun/models/whisper-large-v3",
        #tokenizer = "/models/whisper-large-v3",
        limit_mm_per_prompt={"audio": 1},
        gpu_memory_utilization = 0.8,
        dtype = "float16",
        max_num_seqs = 4,
        max_num_batched_tokens=448,
        # kv_cache_dtype = torch.float8_e4m3fn,
        # trust_remote_code = True,
        # hf_token = True,

    )


audio_files = Path("/app/toqiu/TTS/tests/data/ljspeech/wavs").glob("*.wav")


samples = {}

for file in list(audio_files):
    # Load the audio file
    audio, sample_rate = load(file,sr=16000)
    if sample_rate != 16000:
        # Use librosa to resample the audio
        audio = resample(audio.numpy().astype(np.float32), orig_sr=sample_rate, target_sr=16000)
    print(f"File: {file}, Sample rate: {sample_rate}, Audio shape: {audio.shape}, Duration: {audio.shape[0] / sample_rate:.2f} seconds")
    chunks = chunking(audio, 16000)
    samples[file.stem] = [(chunk,16000) for chunk in chunks]

for file, chunks in samples.items():

    prompts = [{
                "encoder_prompt": {
                    "prompt": "<|startoftranscript|>",
                    "multi_modal_data": {
                        "audio": chunk, # just use the first chunk to debug
                    },
                },
                "decoder_prompt":
                f"<|startoftranscript|>"
            } for chunk in chunks ] *1

    print(f"File: {file}, Chunks: {len(chunks)}")
    # Create a sampling params object.
    sampling_params = SamplingParams(
        temperature=0,
        top_p=1.0,
        max_tokens=8192,
    )

    start = time.time()

    # Inferense based on max_num_seqs
    outputs = []
    #for i in range(0, len(prompts)):
    output = whisper.generate(prompts[0], sampling_params=sampling_params)
    outputs.extend(output)
    # Print the outputs.
    generated = ""
    """for output in outputs:
        prompt = output.prompt
        encoder_prompt = output.encoder_prompt
        generated_text = output.outputs[0].text
        generated+= generated_text
        print(f"Encoder prompt: {encoder_prompt!r}, "
            f"Decoder prompt: {prompt!r}, "
            f"Generated text: {generated_text!r}")
    """
    duration = time.time() - start

    print("Duration:", duration)
    print("RPS:", 1.0 / duration)
    #print("Generated text:", generated)


