import os
os.environ['VLLM_USE_V1'] = '1' 
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
#os.environ['VLLM_LOGGING_LEVEL'] = 'DEBUG'
#os.environ['VLLM_ATTENTION_BACKEND']='FLASH_ATTN' 
os.environ['VLLM_USE_ROCM_AITER'] = '1'
#os.environ['VLLM_ROCM_USE_AITER_MHA'] = '0'
#os.environ['VLLM_ATTENTION_BACKEND'] = 'ROCM_ATTN'
#os.environ['VLLM_ROCM_CUSTOM_PAGED_ATTN'] = '1'
#os.environ['VLLM_ATTENTION_BACKEND']='ROCM_AITER_FA' 
os.environ['VLLM_ATTENTION_BACKEND']='ROCM_AITER_UNIFIED_ATTN' 

use_language_detect = os.getenv("USE_LANGUAGE_DETECT", default="") != ""


# VLLM_USE
LANG_ID_TO_LANG_TOKEN = {
        50259: "<|en|>",
        50260: "<|zh|>",
        50261: "<|de|>",
        50262: "<|es|>",
        50263: "<|ru|>",
        50264: "<|ko|>",
        50265: "<|fr|>",
        50266: "<|ja|>",
        50267: "<|pt|>",
        50268: "<|tr|>",
        50269: "<|pl|>",
        50270: "<|ca|>",
        50271: "<|nl|>",
        50272: "<|ar|>",
        50273: "<|sv|>",
        50274: "<|it|>",
        50275: "<|id|>",
        50276: "<|hi|>",
        50277: "<|fi|>",
        50278: "<|vi|>",
        50279: "<|he|>",
        50280: "<|uk|>",
        50281: "<|el|>",
        50282: "<|ms|>",
        50283: "<|cs|>",
        50284: "<|ro|>",
        50285: "<|da|>",
        50286: "<|hu|>",
        50287: "<|ta|>",
        50288: "<|no|>",
        50289: "<|th|>",
        50290: "<|ur|>",
        50291: "<|hr|>",
        50292: "<|bg|>",
        50293: "<|lt|>",
        50294: "<|la|>",
        50295: "<|mi|>",
        50296: "<|ml|>",
        50297: "<|cy|>",
        50298: "<|sk|>",
        50299: "<|te|>",
        50300: "<|fa|>",
        50301: "<|lv|>",
        50302: "<|bn|>",
        50303: "<|sr|>",
        50304: "<|az|>",
        50305: "<|sl|>",
        50306: "<|kn|>",
        50307: "<|et|>",
        50308: "<|mk|>",
        50309: "<|br|>",
        50310: "<|eu|>",
        50311: "<|is|>",
        50312: "<|hy|>",
        50313: "<|ne|>",
        50314: "<|mn|>",
        50315: "<|bs|>",
        50316: "<|kk|>",
        50317: "<|sq|>",
        50318: "<|sw|>",
        50319: "<|gl|>",
        50320: "<|mr|>",
        50321: "<|pa|>",
        50322: "<|si|>",
        50323: "<|km|>",
        50324: "<|sn|>",
        50325: "<|yo|>",
        50326: "<|so|>",
        50327: "<|af|>",
        50328: "<|oc|>",
        50329: "<|ka|>",
        50330: "<|be|>",
        50331: "<|tg|>",
        50332: "<|sd|>",
        50333: "<|gu|>",
        50334: "<|am|>",
        50335: "<|yi|>",
        50336: "<|lo|>",
        50337: "<|uz|>",
        50338: "<|fo|>",
        50339: "<|ht|>",
        50340: "<|ps|>",
        50341: "<|tk|>",
        50342: "<|nn|>",
        50343: "<|mt|>",
        50344: "<|sa|>",
        50345: "<|lb|>",
        50346: "<|my|>",
        50347: "<|bo|>",
        50348: "<|tl|>",
        50349: "<|mg|>",
        50350: "<|as|>",
        50351: "<|tt|>",
        50352: "<|haw|>",
        50353: "<|ln|>",
        50354: "<|ha|>",
        50355: "<|ba|>",
        50356: "<|jw|>",
        50357: "<|su|>",
        50358: "<|yue|>",
    }

def main():
    import time
    import requests
    import numpy as np
    from datasets import Audio
    from vllm import LLM, SamplingParams
    #from vllm.sampling_params import BeamSearchParams
    from vllm.assets.audio import AudioAsset
    from librosa import resample, load
    sr = 16000
    num_prompts = 256
    audio = Audio(sampling_rate=sr)

    """
    init vllm
    """
    llm = LLM(
        model="/models/whisper-large-v3-turbo",
        max_model_len=448,
        max_num_seqs=256,
        limit_mm_per_prompt={"audio": 1},
        #compilation_config={"level":1}
        # dtype="bfloat16",
        #kv_cache_dtype="fp8",
        #enforce_eager=True
        
    )
    id2token = LANG_ID_TO_LANG_TOKEN
    warmup_sampling_params = SamplingParams(
        temperature=0,
        top_p=1.0,
        max_tokens=1,
        skip_special_tokens = False,
        allowed_token_ids=list(id2token.keys()),
    )
    """
    warm up
    """

    def load_audio(audio_file, target_sr=None):
        import librosa
        """
            audio_arr (np.float32)
            sample_rate
        """
        # sr=None = 用音频原始采样率，不重采样
        audio_arr, sample_rate = librosa.load(audio_file, sr = target_sr)

        # librosa 默认返回 float32 numpy
        return audio_arr.astype(np.float32), sample_rate



    chunks = []
    audio_file = "commonvoice_22_de_test.wav"
    audio_arr, sample_rate = load_audio(audio_file)

    chunks.append((audio_arr, sample_rate))
    warmup_promps = []
    for chunk in chunks:
        warmup_promps.append(
            {
            "prompt": f"<|startoftranscript|>",
            "multi_modal_data": {
                "audio": chunk,
            },
        }
        ) 


    _ = llm.generate(warmup_promps,  warmup_sampling_params)
    # _ = llm.beam_search(PROMPTS[0], params)
    language_token_ids = []
    for output in _:
        generated_text = output.outputs[0].text
        print(f"language token: {output.outputs[0].token_ids[0]}, language type {generated_text!r}")
        language_token_ids.append(output.outputs[0].token_ids[0])
    print("[INFO] warm up ok")

    """
    load audio
    """


    prompts = []
    for language_id, chunk in zip(language_token_ids,chunks):
        if use_language_detect:
            prompts.append({
                    "prompt": f"<|startoftranscript|>{LANG_ID_TO_LANG_TOKEN[language_id]}<|transcribe|><|notimestamps|>",
                    "multi_modal_data": {
                        "audio": chunk,
                    },
                })
        else:
            prompts.append({
                    "prompt": "<|startoftranscript|>",
                    "multi_modal_data": {
                        "audio": chunk,
                    },
                })
    print(prompts)
    sampling_params = SamplingParams(
        temperature=0,
        top_p=1.0,
        max_tokens=500,
    )

    start = time.time()
    outputs = llm.generate(prompts, sampling_params)
    duration = time.time() - start

    for output in outputs:
        generated_text = output.outputs[0].text
        print(f"Generated text: {generated_text!r}")
        #print(output)
        #break



    # print(f"len(prompts):{len(prompts)}")
    # print("Duration:", duration)
    # print("RPS:", len(prompts) / duration)


if __name__ == "__main__":
    main()
