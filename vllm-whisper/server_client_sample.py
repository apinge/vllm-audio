"""
In server part 
use
VLLM_TORCH_PROFILER_DIR="./whisper_profile" CUDA_VISIBLE_DEVICES=7 \
vllm serve "/models/whisper-large-v3" \
--no-enable-chunked-prefill \
--host 0.0.0.0 --port 20010 \

Then use the code in the client
"""
from openai import OpenAI
from vllm.assets.audio import AudioAsset

client = OpenAI(
    api_key="EMPTY",  
    base_url="http://localhost:20010/v1"
)

# Choose your audio file
audio_path = AudioAsset("mary_had_lamb").get_local_path()
with open(audio_path, "rb") as f:
    resp = client.audio.transcriptions.create(
        file=f,
        model="/models/whisper-large-v3",
        language="en",
        response_format="text",  # or "json"
        temperature=0.0
    )
print(resp)
