# SPDX-License-Identifier: Apache-2.0
"""
This example shows how to use vLLM for running offline inference 
with the correct prompt format on audio language models.

For most models, the prompt format should follow corresponding examples
on HuggingFace model repository.
"""
from transformers import AutoTokenizer

from vllm import LLM, SamplingParams
from vllm.assets.audio import AudioAsset
from vllm.utils import FlexibleArgumentParser

audio_assets = [AudioAsset("mary_had_lamb"), AudioAsset("winning_call")]
question_per_audio_count = {
    0: "What is 1+1?",
    1: "What is recited in the audio?",
    2: "What sport and what nursery rhyme are referenced?"
}

# NOTE: The default `max_num_seqs` and `max_model_len` may result in OOM on
# lower-end GPUs.
# Unless specified, these settings have been tested to work on a single L4.


# ============== real audio data folder ==============
#AUDIO_FOLDER = "/root/workspace/tts-root/TTS/tests/data/ljspeech/wavs" #32
AUDIO_FOLDER = "/root/workspace/audio_dataset/LibriSpeech/dev-clean" #2703
import os
import glob
import librosa

def load_audio_folder(folder_path, sr=None, exts=("wav", "flac")):
    """Load all *.wav or *.flac files, return [(waveform numpy array, sample_rate), ...]"""
    audio_items = []
    for ext in exts:
        for file_path in glob.glob(os.path.join(folder_path, "**", f"*.{ext}"), recursive=True):
            try:
                audio, sample_rate = librosa.load(file_path, sr=sr) # sr=None uses the native sampling rate
                audio_items.append((audio, sample_rate))
            except Exception as e:
                print(f"Load {file_path} failure: {e}")
    print(f"load {len(audio_items)} files")
    return audio_items

# Qwen2-Audio
def get_qwen2_audio_prompts(question: str, audio_count: int):
    

    
    audio_in_prompt = "".join([
        f"Audio {idx+1}: "
        f"<|audio_bos|><|AUDIO|><|audio_eos|>\n" for idx in range(audio_count)
    ])

    prompt = ("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
              "<|im_start|>user\n"
              f"{audio_in_prompt}{question}<|im_end|>\n"
              "<|im_start|>assistant\n")
    stop_token_ids = None
    return prompt, stop_token_ids




model_example_st = {
    "qwen2_audio"
}


def main(args):
    model = args.model_type
    if model not in model_example_st:
        raise ValueError(f"Model type {model} is not supported.")
    audio_count = args.num_audios #1

    prompt, stop_token_ids = get_qwen2_audio_prompts(
    question_per_audio_count[audio_count], audio_count)
    all_audios = load_audio_folder(AUDIO_FOLDER, sr=None)
    if len(all_audios) < args.num_prompts:
        raise ValueError(f"Insufficient number of audio files: required at least{audio_count}, actual len{len(all_audios)}")
    inputs = []
    warm_up_steps = 8
    for i in range(args.num_prompts+warm_up_steps): 

        mm_data = {
            "audio": [
                all_audios[i]
                # asset.audio_and_sample_rate
                # for asset in audio_assets[:audio_count]
            ]
        }

    # assert args.num_prompts > 0
        base_input = {"prompt": prompt, "multi_modal_data": mm_data}
        inputs.append(base_input)
    # Init llm
    model_name = "/models/Qwen2-Audio-7B-Instruct"
    llm = LLM(model=model_name,
              max_model_len=4096,
              max_num_seqs=64,
              limit_mm_per_prompt={"audio": audio_count},
              tensor_parallel_size=1,
              compilation_config = {"full_cuda_graph":True},
              enable_prefix_caching=False, # set this to ensure accurate measurement
              )


    
 

    # We set temperature to 0.2 so that outputs can be different
    # even when all prompts are identical when running batch inference.
    sampling_params = SamplingParams(temperature=0.2,
                                     max_tokens=32,
                                     stop_token_ids=stop_token_ids)

    # warm up
    _outputs = llm.generate(inputs[:warm_up_steps], sampling_params=sampling_params)
    for o in _outputs:
        generated_text = o.outputs[0].text
        print(generated_text)
    print("Warm up steps completed.")

    assert args.num_prompts == len(inputs[warm_up_steps:])
    import time
    start = time.time()
    outputs = llm.generate(inputs[warm_up_steps:], sampling_params=sampling_params)
    duration = time.time() - start
    print("Duration:", duration)
    print("RPS:", args.num_prompts / duration)
    total_tokens = 0
    for o in outputs:
        generated_text = o.outputs[0].text
        total_tokens += len(generated_text.split())
        #print(generated_text)
    print(f"Total tokens, {total_tokens}")


if __name__ == "__main__":
    parser = FlexibleArgumentParser(
        description='Demo on using vLLM for offline inference with '
        'audio language models')
    parser.add_argument('--model-type',
                        '-m',
                        type=str,
                        default="qwen2_audio",
                        choices=model_example_st,
                        help='Huggingface "model_type".')
    parser.add_argument('--num-prompts',
                        type=int,
                        default=512,
                        help='Number of prompts to run.')
    parser.add_argument("--num-audios",
                        type=int,
                        default=1,
                        choices=[0, 1, 2],
                        help="Number of audio items per prompt.")

    args = parser.parse_args()
    main(args)
