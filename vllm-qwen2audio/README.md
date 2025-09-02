# How to Run
```bash
docker pull rocm/vllm:rocm6.4.1_vllm_0.10.0_20250812

docker run -it   --name qwen2audio --ipc=host --network=host --group-add render --privileged --security-opt seccomp=unconfined --cap-add=CAP_SYS_ADMIN --cap-add=SYS_PTRACE --device=/dev/kfd --device=/dev/dri --device=/dev/mem  -v /raid/users/toqiu:/root/workspace   -v /raid/models:/models  rocm/vllm:rocm6.4.1_vllm_0.10.0_20250812

pip install vllm[audio]
```

Test code refers to [Audio Language](https://docs.vllm.ai/en/v0.7.3/getting_started/examples/audio_language.html)
