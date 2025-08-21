#!/usr/bin/env python
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0


from vllm import LLM, SamplingParams
from vllm.assets.audio import AudioAsset

# Simple example to test whisper according to unit test
MODEL_NAME = "/models/whisper-large-v3"

PROMPTS = [
    {
        "prompt": "<|startoftranscript|><|en|><|transcribe|><|notimestamps|>",
        "multi_modal_data": {
            "audio": AudioAsset("mary_had_lamb").audio_and_sample_rate,
        },
    },

]

def main():

    llm = LLM(
        MODEL_NAME,
        dtype="half",
        max_model_len=448,
        tensor_parallel_size=1,  
        gpu_memory_utilization = 0.5
    )

    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=200,
    )

    outputs = llm.generate(PROMPTS, sampling_params)

    for i, output in enumerate(outputs):
        print(f"\n==== Sample {i} ====")
        print(output.outputs[0].text)

if __name__ == "__main__":
    main()
