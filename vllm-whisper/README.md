# How to Run
Use the code from https://github.com/Drakrig/whisper_vllm_example/blob/main/step_by_step.ipynb

```bash
docker run -it --rm  --name whisper --ipc=host --network=host --group-add render --privileged --security-opt seccomp=unconfined --cap-add=CAP_SYS_ADMIN --cap-add=SYS_PTRACE --device=/dev/kfd --device=/dev/dri --device=/dev/mem -v /home/taccuser/:/app/  vllm-rocm_v0.10.1:latest bash

pip install vllm[audio]

```
