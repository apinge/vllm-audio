from openai import OpenAI
import random
import os
client = OpenAI(base_url="http://localhost:30000/v1", api_key="None")

# Construct image_url：imgset1.jpg -> imgset13.jpg
image_contents = []
for i in range(1, 14):
    path = f"/root/workspace/qwen_omni_sglang/imgset{i}.jpg"
    if os.path.isfile(path):
        image_contents.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"/root/workspace/qwen_omni_sglang/imgset{i}.jpg",
                },
            }
        )
    else:
        print(f"{path} is missing")
    
#random.shuffle(image_contents)
# print("num images:", len(image_contents))
# print([c["image_url"]["url"] for c in image_contents])

audio_content = {
    "type": "audio_url",
    "audio_url": {
        "url": "/root/workspace/qwen_omni_sglang/new_audio.wav",
    },
}

text_content = {
                    "type": "text",
                    "text": "Describe each image in chinese separately."
                }

messages = [
    {
        "role": "user",
        "content": image_contents + [audio_content]+[text_content] ,
    }
]



response = client.chat.completions.create(
    model="/models/Qwen3-Omni-30B-A3B-Instruct",
    messages=messages,
    temperature=0,
    max_tokens=2048,
    #stream=True,
)

print(response.choices[0].message.content)
