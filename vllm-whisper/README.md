# How to Run
Use the code from https://github.com/Drakrig/whisper_vllm_example/blob/main/step_by_step.ipynb

The base docker used is https://hub.docker.com/layers/rocm/vllm/rocm6.4.1_vllm_0.10.0_20250812/images/sha256-4c277ad39af3a8c9feac9b30bf78d439c74d9b4728e788a419d3f1d0c30cacaa
```bash
docker run -it --name whisper --ipc=host --network=host --group-add render --privileged --security-opt seccomp=unconfined --cap-add=CAP_SYS_ADMIN --cap-add=SYS_PTRACE --device=/dev/kfd --device=/dev/dri --device=/dev/mem -v /home/taccuser/:/app/  vllm-rocm_v0.10.1:latest bash

pip install vllm[audio]

```

If you want translate task you can specify the task
```python
    prompts = [
    {
        "prompt":
        "<|startoftranscript|><|ar|><|translate|><|notimestamps|>",
        "multi_modal_data": {
            "audio": chunk, # Audio Chunk
        },
    },
```

# How to Run using VLLM Benchmark
```bash
CUDA_VISIBLE_DEVICES=7 vllm serve "/models/whisper-large-v3" --no-enable-chunked-prefill --disable-log-requests --host 0.0.0.0 --port 20010
 
python benchmarks/benchmark_serving.py --model "/models/whisper-large-v3" --backend openai-audio --dataset-name hf  --dataset-path edinburghcstr/ami --hf-subset ihm --endpoint /v1/audio/transcriptions --trust_remote_code  --num-prompts 16 --max-concurrency 4 --host 0.0.0.0 --port 20010

```
