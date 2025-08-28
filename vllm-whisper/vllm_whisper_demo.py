from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import time
from vllm import LLM, SamplingParams
from transformers import WhisperTokenizerFast
from pathlib import Path
from librosa import resample, load
import numpy as np


def chunking(audio: np.ndarray, sample_rate: int):
    """Split audio to 30 second duration chunks

    Args:
        audio (np.ndarray): Audio data
        sample_rate (int): Audio sample rate
    """
    max_duration_samples = sample_rate * 30.0
    padding = max_duration_samples - np.remainder(len(audio), max_duration_samples)
    audio = np.pad(audio, (0, padding.astype(int)), "constant", constant_values=0.0)
    return np.split(audio, len(audio) // max_duration_samples)


tokenizer = WhisperTokenizerFast.from_pretrained(
    "/models/whisper-large-v3", language="en"
)
whisper = LLM(
    model="/models/whisper-large-v3",
    limit_mm_per_prompt={"audio": 1},
    gpu_memory_utilization=0.8,
    dtype="float16",
    max_num_seqs=4,
    max_num_batched_tokens=448,
)


audio_files = Path("samples").glob("*.wav")


samples = {}

for file in list(audio_files):
    # Load the audio file
    audio, sample_rate = load(file, sr=16000)
    if sample_rate != 16000:
        # Use librosa to resample the audio
        audio = resample(
            audio.numpy().astype(np.float32), orig_sr=sample_rate, target_sr=16000
        )
    print(
        f"File: {file}, Sample rate: {sample_rate}, Audio shape: {audio.shape}, Duration: {audio.shape[0] / sample_rate:.2f} seconds"
    )
    chunks = chunking(audio, 16000)
    samples[file.stem] = [(chunk, 16000) for chunk in chunks]

for file, chunks in samples.items():
    prompts = [
        {
            "prompt": "<|startoftranscript|><|en|><|transcribe|><|notimestamps|>",
            "multi_modal_data": {
                "audio": chunk,
            },
        }
        for chunk in chunks
    ]
    print(f"File: {file}, Chunks: {len(chunks)}")
    # Create a sampling params object.
    sampling_params = SamplingParams(
        temperature=0,
        top_p=1.0,
        max_tokens=8192,
    )

    start = time.time()

    # Inferense based on max_num_seqs
    outputs = whisper.generate(prompts, sampling_params=sampling_params)

    # Print the outputs.
    generated = ""
    # prompt = ""
    for output in outputs:
        # prompt = output.prompt
        # encoder_prompt = output.encoder_prompt
        generated_text = output.outputs[0].text
        generated += generated_text

    duration = time.time() - start
    print("====================================")
    print("Duration:", duration)
    print("RPS:", len(prompts) / duration)
    print("Generated text:", generated)
