import os
os.environ['VLLM_USE_V1'] = '0'
import torch


from transformers import Qwen3OmniMoeProcessor
from qwen_omni_utils import process_mm_info
from vllm import LLM, SamplingParams
import librosa



import os
import torch
import librosa
from typing import List, Tuple, Dict, Any



def prepare_audio_inputs_for_vllm(audio_folder, processor, use_audio_in_video=False):
    filenames = []
    inputs_list = []

    for filename in sorted(os.listdir(audio_folder)):
        if not filename.lower().endswith(".wav"):
            continue
     
        
        file_path = os.path.join(audio_folder, filename)
        y, sr = librosa.load(file_path, sr=None)
        duration = len(y) / sr
        if duration>30:
            print(f"{file_path} audio length: {duration:.2f} second")
        else:
            continue # skip small audio
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": file_path},
                    {
                    "text": '\nAnalyze this audio and generate a reasoning chain to deduce the content of the audio. The specific steps are as follows:\n\n1. Generate a reasoning chain for the description of the audio:\n* Determine which type the audio belongs to (Music / Sound / Speech / Song). The distinction between speech and sound is whether or not it contains intelligible human voice content. The difference between music and a song lies in that a song contains vocal content, while music does not.\n* If the audio is **Speech**, then analyze the following aspects of the audio: (1) Spoken Language Characteristic, including language, number of speakers, speaker gender, speaker emotion, and sentiment; (2) Speech Transcript, which refers to the textual content of the speech; (3) Sound Caption, which is a description of this audio, with particular emphasis on describing the background sounds of the speech.\n* If the audio is **Music**, then analyze the following aspects of the audio: (1) Music Reasoning, including Genre and Style, Mood and Expression; (2) Music Knowledge, including types of instruments, sound texture, melody and rhythm, harmony, and chords; (3) Historical and Cultural Context, which involves analyzing the style of the audio, its background, and information about the author\'s style.\n* If the audio is **Sound**, then analyze the following aspects of the audio: (1) Acoustic Sounds analysis, which means analyzing the conditions of the audio, such as weather phenomena, wild environments, non-verbal vocalizations etc.; (2) Acoustic Scene Reasoning, which involves describing the scene of the sound and inferring the sound events and activities reflected by the related audio; (3) Sound-Based Event Reasoning, which involves describing repeated sounds, fixed frequency sounds, and regular sound phenomena. \n* If the audio is **Song**, then analyze the following aspects of the audio: (1) Song Reasoning, including Genre and Style, Mood and Expression, a summary of the lyrics, specific lyrics; (2) Song Knowledge, including types of instruments, sound texture, melody and rhythm, harmony, and chords; (3) Historical and Cultural Context, which involves analyzing the style of the audio, its background, and information about the author\'s style.\n\n2. Please integrate all of the above results in JSON format. Below is a sample:\n```json\n{{\n    "type": "",\n    "content_dict": {{ // follow the various items for different types\n    //...\n    }}\n}}\n',
                    "type": "text",
                },
                ],
            }
        ]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        audios, _, _ = process_mm_info(messages, use_audio_in_video=False)

        inputs = {"prompt": text, "multi_modal_data": {}}
        if audios is not None:
            inputs["multi_modal_data"]["audio"] = audios

        inputs_list.append(inputs)
        filenames.append(filename)

    if not inputs_list:
        raise ValueError(f"No .wav files found in {audio_folder}")

    print(f"[INFO] Prepared {len(inputs_list)} inputs from {audio_folder}")
    return inputs_list, filenames


#from vllm.config import CompilationConfig, CompilationLevel
if __name__ == '__main__':
    # vLLM engine v1 not supported yet
    MODEL_PATH = "/models/Qwen3-Omni-30B-A3B-Captioner"
    audio_folder = "/root/workspace/qwen_omni/1000/"
    processor = Qwen3OmniMoeProcessor.from_pretrained(MODEL_PATH)
    inputs_list, filenames = prepare_audio_inputs_for_vllm(
        audio_folder=audio_folder,
        processor=processor,
        use_audio_in_video=False
    )
    if len(inputs_list)>512:
        inputs_list = inputs_list[:512]
        print(f"[WARNNING] Only take {len(inputs_list)} prompts")

    llm = LLM(
        model=MODEL_PATH,
        trust_remote_code=True,
        gpu_memory_utilization=0.95,
        tensor_parallel_size=torch.cuda.device_count(),
        limit_mm_per_prompt={"audio": 1},
        max_num_seqs=8, #only 8
        max_model_len=32768,
        seed=1234,
    )

    sampling_params = SamplingParams(
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        max_tokens=16384,
    )
    print("[INFO] load model succeed.")
    outputs = llm.generate(inputs_list, sampling_params=sampling_params)
    for filename, req_output in zip(filenames, outputs):
        completion_output = req_output.outputs[0]
        print(f"\n=== Output for {filename} ===")
        print(completion_output.text)
        m = req_output.metrics
        print(f"TTFT(end-to-end): {m.first_token_time - m.arrival_time:.3f} second")
        print(
            f"TTFT(model-only): {m.first_token_time - m.first_scheduled_time:.3f} second"
        )