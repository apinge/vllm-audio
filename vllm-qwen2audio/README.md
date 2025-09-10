# How to Run
```bash
docker pull rocm/vllm:rocm6.4.1_vllm_0.10.0_20250812

docker run -it   --name qwen2audio --ipc=host --network=host --group-add render --privileged --security-opt seccomp=unconfined --cap-add=CAP_SYS_ADMIN --cap-add=SYS_PTRACE --device=/dev/kfd --device=/dev/dri --device=/dev/mem  -v /raid/users/toqiu:/root/workspace   -v /raid/models:/models  rocm/vllm:rocm6.4.1_vllm_0.10.0_20250812

pip install vllm[audio]
```

Test code refers to [Audio Language](https://docs.vllm.ai/en/v0.7.3/getting_started/examples/audio_language.html)

## About the docker

Use the new docker
```bash
docker run -it --rm --ipc=host --network=host --group-add video --privileged --security-opt seccomp=unconfined --cap-add=CAP_SYS_ADMIN --cap-add=SYS_PTRACE --device=/dev/kfd --device=/dev/dri --device=/dev/mem -e VLLM_ROCM_USE_AITER=1 -e VLLM_ROCM_USE_AITER_RMSNORM=1 -e VLLM_USE_TRITON_FLASH_ATTN=0 -e VLLM_ROCM_USE_AITER_MHA=1 -e VLLM_ROCM_USE_AITER_LINEAR=1 -e VLLM_ROCM_CUSTOM_PAGED_ATTN=0 -e VLLM_ROCM_USE_AITER_PAGED_ATTN=1 -e HIP_FORCE_DEV_KERNARG=1 -v /raid/models/:/app rocm/7.0-preview:rocm7.0_preview_ubuntu_22.04_vllm_0.10.1_instinct_beta
```

# How to use Qwen audio with vllm Benchmark
Currently I add some workaround to enable the audio encoder when using vllm benchmark.

In file `/opt/venv/lib/python3.10/site-packages/vllm/entrypoints/chat_utils.py` , modify the function like
```python
def parse_chat_messages_futures(
    messages: list[ChatCompletionMessageParam],
    model_config: ModelConfig,
    tokenizer: AnyTokenizer,
    content_format: _ChatTemplateContentFormat,
) -> tuple[list[ConversationMessage], Awaitable[Optional[MultiModalDataDict]]]:

    conversation: list[ConversationMessage] = []

    mm_tracker = AsyncMultiModalItemTracker(model_config, tokenizer)

    for msg in messages:
        audio_part ={
            "type": "audio_url",
          "audio_url": {
            "url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/audio/guess_age_gender.wav"
          }
        }
        msg['content'].insert(len(msg['content']), audio_part)
```

In server
```bash
MIOPEN_FIND_MODE=FAST MIOPEN_USER_DB_PATH="./miopen_cache"  VLLM_TORCH_PROFILER_DIR="./profile" CUDA_VISIBLE_DEVICES=6 VLLM_ROCM_USE_AITER=1  VLLM_ROCM_USE_AITER_RMSNORM=1 VLLM_USE_TRITON_FLASH_ATTN=0A=1 VLLM_ROCM_USE_AITER_LINEAR=1  VLLM_ROCM_CUSTOM_PAGED_ATTN=0 VLLM_ROCM_USE_AITER_PAGED_ATTN=1 HIP_FORCE_DEV_KERNARG=1 vllm serve /models/Qwen2-Audio-7B-Instruct/ --served-model-name qwen2-audio-7b --tensor_parallel_size=1 --disable-log-requests  --dtype bfloat16  --no-enable-prefix-caching --no-enable-chunked-prefill
```
In client (use `--backend openai-chat` and `--endpoint /v1/chat/completions` to enable the audio data with chat)

```bash
python benchmarks/benchmark_serving.py --backend openai-chat  --model qwen2-audio-7b --tokenizer /models/Qwen2-Audio-7B-Instruct/   --endpoint /v1/chat/completions --dataset-name random --random-input-len 1024 --random-output-len 10 --ignore-eos --num-prompts 8 --max-concurrency 8 --base-url http://127.0.0.1:8000 --host 0.0.0.0 --port 8000 --percentile-metrics ttft,tpot,itl,e2el
```


If you want to profile, then add `--profile` in the client command line.



