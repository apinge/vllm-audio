from openai import OpenAI
import random
import os
import base64
import numpy as np
import librosa
from io import BytesIO
import soundfile as sf

client = OpenAI(base_url="http://localhost:40000/v1", api_key="None")


def file_to_base64(file_path):
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def create_image_content(images):
    return [
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{file_to_base64(img)}"},
        }
        for img in images
    ]


# https://github.com/sgl-project/sglang/pull/12775
def encode_audio(audio_array: np.ndarray, sampling_rate: int) -> str:
    """Encodes a NumPy audio array into a base64 string."""
    buffered = BytesIO()
    sf.write(buffered, audio_array, sampling_rate, format="WAV")
    audio_bytes = buffered.getvalue()
    return base64.b64encode(audio_bytes).decode("utf-8")


# Construct image_url：imgset1.jpg -> imgset13.jpg
image_list = []
for i in range(1, 14):
    path = f"/root/workspace/qwen_omni_sglang/imgset{i}.jpg"
    if os.path.isfile(path):
        image_list.append(
            f"/root/workspace/qwen_omni_sglang/imgset{i}.jpg",
        )
    else:
        print(f"{path} is missing")

image_content = create_image_content(image_list)
# random.shuffle(image_contents)
# print("num images:", len(image_contents))
# print([c["image_url"]["url"] for c in image_contents])

"""audio_content = {
    "type": "audio_url",
    "audio_url": {
        "url": "/root/workspace/qwen_omni_sglang/new_audio.wav",
    },
}"""

audio_path = "/root/workspace/qwen_omni_sglang/new_audio.wav"
y, sr = librosa.load(audio_path, sr=None)

audio_base64 = encode_audio(audio_array=y, sampling_rate=sr)
audio_content = {
    "type": "audio_url",
    "audio_url": {"url": f"data:audio/wav;base64,{audio_base64}"},
}


text_content = {
    "type": "text",
     "text": "Describe each image in chinese separately."
    #"text": "Describe what you have heard in audio.", # text asr
}

messages = [
    {
        "role": "user",
        "content": image_content + [audio_content],
        # "content":  [audio_content] + [text_content], # text asr
    }
]


response = client.chat.completions.create(
    model="/models/Qwen3-Omni-30B-A3B-Instruct",
    messages=messages,
    temperature=0,
    max_tokens=2048,
    stream=True,
)

# print(response.choices[0].message.content)
for chunk in response:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="")
