# How to Run
Use the code from https://github.com/Drakrig/whisper_vllm_example/blob/main/step_by_step.ipynb

The base docker used is https://hub.docker.com/layers/rocm/vllm/rocm6.4.1_vllm_0.10.0_20250812/images/sha256-4c277ad39af3a8c9feac9b30bf78d439c74d9b4728e788a419d3f1d0c30cacaa
```bash
docker run -it --rm  --name whisper --ipc=host --network=host --group-add render --privileged --security-opt seccomp=unconfined --cap-add=CAP_SYS_ADMIN --cap-add=SYS_PTRACE --device=/dev/kfd --device=/dev/dri --device=/dev/mem -v /home/taccuser/:/app/  vllm-rocm_v0.10.1:latest bash

pip install vllm[audio]

```
