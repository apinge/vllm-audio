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
    # Init llm
    model_name = "/models/Qwen2-Audio-7B-Instruct"
    llm = LLM(model=model_name,
              max_model_len=4096,
              max_num_seqs=16,
              limit_mm_per_prompt={"audio": audio_count})


    
    prompt, stop_token_ids = get_qwen2_audio_prompts(
        question_per_audio_count[audio_count], audio_count)

    # We set temperature to 0.2 so that outputs can be different
    # even when all prompts are identical when running batch inference.
    sampling_params = SamplingParams(temperature=0.2,
                                     max_tokens=32,
                                     stop_token_ids=stop_token_ids)

    mm_data = {}
    if audio_count > 0:
        mm_data = {
            "audio": [
                asset.audio_and_sample_rate
                for asset in audio_assets[:audio_count]
            ]
        }

    assert args.num_prompts > 0
    base_input = {"prompt": prompt, "multi_modal_data": mm_data}
    if args.num_prompts > 1:
        # Batch inference
        import copy
        inputs = [copy.deepcopy(base_input) for _ in range(args.num_prompts)]
    else:
        inputs = base_input
    import time
    start = time.time()
    outputs = llm.generate(inputs, sampling_params=sampling_params)
    duration = time.time() - start
    print("Duration:", duration)
    print("RPS:", args.num_prompts / duration)
    # for o in outputs:
    #     generated_text = o.outputs[0].text
    #     print(generated_text)


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
                        default=64,
                        help='Number of prompts to run.')
    parser.add_argument("--num-audios",
                        type=int,
                        default=1,
                        choices=[0, 1, 2],
                        help="Number of audio items per prompt.")

    args = parser.parse_args()
    main(args)
