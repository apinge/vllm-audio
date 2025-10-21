# How to Use V0 version

```bash
docker run -it  --name qwen-omni-vllm-master-ubuntu-dev- --ipc=host --network=host  --privileged --security-opt seccomp=unconfined --cap-add=CAP_SYS_ADMIN --cap-add=SYS_PTRACE --device=/dev/kfd --device=/dev/dri --device=/dev/mem  -v /raid/users/toqiu:/root/workspace  -v /raid/models:/models  rocm/vllm:rocm6.4.1_vllm_0.10.1_20250909

pip uninstall -y vllm

git clone https://github.com/apinge/vllm.git -b qwen3_omni vllm-dev
cd vllm-dev
pip install --upgrade numba \
    scipy \
    huggingface-hub[cli,hf_transfer] \
    setuptools_scm
pip uninstall -y opencv-python-headless
pip install "numpy<2"
pip install -r requirements/rocm.txt

export PYTORCH_ROCM_ARCH="gfx942"
python3 setup.py develop

pip install git+https://github.com/huggingface/transformers
pip install accelerate
pip install qwen-omni-utils -U

apt update -y && apt install ffmpeg -y
# GOTO YOUR WORKSPACE
# run single audio url
python qwen_omni_v0_url.py

# run single audio file
python qwen_omni_v0_readfile.py


# run multiple audio from a folder
python qwen_omni_v0_folder.py

# test audio longer than 30s 
python qwen_omni_v0_folder_test_long_audio.py

```

Other usage please refer to https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Captioner#vllm-usage
