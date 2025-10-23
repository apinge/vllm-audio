import os
os.environ['VLLM_USE_V1'] = '1'
os.environ['VLLM_TORCH_PROFILER_RECORD_SHAPES'] = '1'
os.environ['VLLM_TORCH_PROFILER_WITH_STACK'] = '1'
#os.environ['VLLM_ROCM_USE_AITER'] = '1'
#os.environ['VLLM_ROCM_USE_AITER_MOE'] = '1'
# os.environ['VLLM_ROCM_USE_AITER_LINEAR'] = '1'
# os.environ['VLLM_ROCM_USE_AITER_PAGED_ATTN'] = '1'
# os.environ['VLLM_ROCM_USE_AITER_MHA'] = '1'
VLLM_USE_V1 = os.environ['VLLM_USE_V1']

with_torch_profiler_dir = os.getenv("VLLM_TORCH_PROFILER_DIR", default="") != ""
batch_size = 1
print(f"{with_torch_profiler_dir=} {VLLM_USE_V1=}")

import torch


from transformers import Qwen3OmniMoeProcessor
from qwen_omni_utils import process_mm_info
from vllm import LLM, SamplingParams
#from vllm.config import CompilationConfig, CompilationLevel
if __name__ == '__main__':
    # vLLM engine v1 not supported yet


    MODEL_PATH = "/models/Qwen3-Omni-30B-A3B-Captioner"

    llm = LLM(
            model=MODEL_PATH, trust_remote_code=True, gpu_memory_utilization=0.95,
            tensor_parallel_size=1, #torch.cuda.device_count(),
            limit_mm_per_prompt={'audio': 1},
            max_num_seqs=8,
            max_model_len=32768,
            seed=1234,
            #enforce_eager=True,
            disable_log_stats = False
    )

    sampling_params = SamplingParams(
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        max_tokens= 4 if with_torch_profiler_dir else 320,
    )
    print("[INFO] ok")
    processor = Qwen3OmniMoeProcessor.from_pretrained(MODEL_PATH)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "audio", 
                 "audio": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/cookbook/caption2.mp3"
                 },
                {
                    "text": '\nAnalyze this audio and generate a reasoning chain to deduce the content of the audio. The specific steps are as follows:\n\n1. Generate a reasoning chain for the description of the audio:\n* Determine which type the audio belongs to (Music / Sound / Speech / Song). The distinction between speech and sound is whether or not it contains intelligible human voice content. The difference between music and a song lies in that a song contains vocal content, while music does not.\n* If the audio is **Speech**, then analyze the following aspects of the audio: (1) Spoken Language Characteristic, including language, number of speakers, speaker gender, speaker emotion, and sentiment; (2) Speech Transcript, which refers to the textual content of the speech; (3) Sound Caption, which is a description of this audio, with particular emphasis on describing the background sounds of the speech.\n* If the audio is **Music**, then analyze the following aspects of the audio: (1) Music Reasoning, including Genre and Style, Mood and Expression; (2) Music Knowledge, including types of instruments, sound texture, melody and rhythm, harmony, and chords; (3) Historical and Cultural Context, which involves analyzing the style of the audio, its background, and information about the author\'s style.\n* If the audio is **Sound**, then analyze the following aspects of the audio: (1) Acoustic Sounds analysis, which means analyzing the conditions of the audio, such as weather phenomena, wild environments, non-verbal vocalizations etc.; (2) Acoustic Scene Reasoning, which involves describing the scene of the sound and inferring the sound events and activities reflected by the related audio; (3) Sound-Based Event Reasoning, which involves describing repeated sounds, fixed frequency sounds, and regular sound phenomena. \n* If the audio is **Song**, then analyze the following aspects of the audio: (1) Song Reasoning, including Genre and Style, Mood and Expression, a summary of the lyrics, specific lyrics; (2) Song Knowledge, including types of instruments, sound texture, melody and rhythm, harmony, and chords; (3) Historical and Cultural Context, which involves analyzing the style of the audio, its background, and information about the author\'s style.\n\n2. Please integrate all of the above results in JSON format. Below is a sample:\n```json\n{{\n    "type": "",\n    "content_dict": {{ // follow the various items for different types\n    //...\n    }}\n}}\n',
                    "type": "text",
                },
            ], 
        }
    ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    audios, _, _ = process_mm_info(messages, use_audio_in_video=False)

    inputs = {
        'prompt': text,
        'multi_modal_data': {},
    }

    if audios is not None:
        inputs['multi_modal_data']['audio'] = audios

    batched_inputs = [inputs]*batch_size
    # Only warm up for profile
    if with_torch_profiler_dir: 
        #warm_up
        _ = llm.generate(batched_inputs, sampling_params=sampling_params)
        print("[INFO] warm up completed")

    if with_torch_profiler_dir: llm.start_profile()
    import time
    start_time = time.perf_counter()
    outputs = llm.generate(batched_inputs, sampling_params=sampling_params)
    end_time = time.perf_counter()
    if with_torch_profiler_dir: llm.stop_profile()
    print(f"[E2E] End-to-end latency: {end_time - start_time:.3f} seconds")
    for req_output in outputs:
        completion_output = req_output.outputs[0]
        print("\n=== model output ===")
        print(completion_output.text)
    try:
        metrics = llm.get_metrics()

        def extract_avg(hist_name):
            for m in metrics:
                if m.name == hist_name:
                    print(f"{m.name}, {m.count}")
                    return m.sum / m.count if m.count > 0 else None
            return None

        print("TTFT (s):", extract_avg("vllm:time_to_first_token_seconds"))
        print("Time per token (s):", extract_avg("vllm:time_per_output_token_seconds"))
        print("E2E latency (s):", extract_avg("vllm:e2e_request_latency_seconds"))

    except AssertionError as e:
        print(f"[WARN] Cannot get metrics（maybe disable_log_stats=True？）: {e}")

    except Exception as e:
        import traceback
        print("[ERROR] get metrics failure：")
        traceback.print_exc()
