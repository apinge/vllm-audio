import time
import requests
from datasets import Audio
from vllm import LLM, SamplingParams
from vllm.assets.audio import AudioAsset
from librosa import resample, load
# Create a Whisper encoder/decoder model instance
sr = 16000
num_prompts = 8
audio = Audio(sampling_rate=sr)
#print(AudioAsset("winning_call").url)


"""
init vllm
"""
llm = LLM(
        model="/models/whisper-large-v3",
        #model="/app/models/whisper-large-v3-FP8-Dynamic",
        max_model_len=448,
        max_num_seqs=256,
        limit_mm_per_prompt={"audio": 1},
        #dtype="bfloat16",
        #kv_cache_dtype="fp8",
)
# Create a sampling params object.
sampling_params = SamplingParams(
    temperature=0,
    top_p=1.0,
    max_tokens=500,
)
 

"""
warm up
"""

warmup_prompts = [
    {
        "prompt": "<|startoftranscript|>",
        "multi_modal_data": {
            "audio": AudioAsset("mary_had_lamb").audio_and_sample_rate,
        },
    },
    #"decoder_prompt": "<|startoftranscript|>",
    {  # Test explicit encoder/decoder prompt
        "encoder_prompt": {
            "prompt": "",
            "multi_modal_data": {
                "audio": AudioAsset("winning_call").audio_and_sample_rate,
            },
        },
        "decoder_prompt": "<|startoftranscript|>",
    }]

_  = llm.generate(warmup_prompts,sampling_params)

print("[INFO] warm up ok")
"""
load audio
"""

audio_file = "out.wav"
audio, sample_rate = load(audio_file,sr=None)
if sample_rate != sr:
    # Use librosa to resample the audio
    audio = resample(audio.numpy().astype(np.float32), orig_sr=sample_rate, target_sr=16000)
    print(f"File: {file}, Sample rate: {sample_rate}, Audio shape: {audio.shape}, Duration: {audio.shape[0] / sample_rate:.2f} seconds")
chunk = (audio, sr)


prompts = [
    {
        "prompt": "<|startoftranscript|>",
        "multi_modal_data": {
            "audio": chunk,#AudioAsset("mary_had_lamb").audio_and_sample_rate,
        },
    },
    # #"decoder_prompt": "<|startoftranscript|>",
    # {  # Test explicit encoder/decoder prompt
    #     "encoder_prompt": {
    #         "prompt": "",
    #         "multi_modal_data": {
    #             "audio": AudioAsset("winning_call").audio_and_sample_rate,
    #         },
    #     },
    #     "decoder_prompt": "<|startoftranscript|>",
    # }
]*num_prompts


 
#r = requests.get('https://github.com/mesolitica/malaya-speech/raw/master/speech/7021-79759-0004.wav')
#y = audio.decode_example(audio.encode_example(r.content))['array']
 

start = time.time()
 
# Generate output tokens from the prompts. The output is a list of
# RequestOutput objects that contain the prompt, generated
# text, and other information.
#llm.start_profile()
 
outputs = llm.generate(prompts,sampling_params)

#llm.end_profile()
duration = time.time() - start 
# Print the outputs.
for output in outputs:
    prompt = output.prompt
    #encoder_prompt = output.encoder_prompt
    generated_text = output.outputs[0].text
    #print(f"Generated text: {generated_text!r}")
 

print(f"len(prompts):{len(prompts)}")
print("Duration:", duration)
print("RPS:", len(prompts) / duration)

 
 
