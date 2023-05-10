import runhouse as rh
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def causal_lm_generate(prompt, model_id='google/flan-t5-xl', **model_kwargs):
    (tokenizer, model) = rh.get_pinned_object(model_id) or (None, None)
    if model is None:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_id).to('cuda')
        rh.pin_to_memory(model_id, (tokenizer, model))
    inputs = tokenizer(prompt, return_tensors="pt").to('cuda')
    outputs = model.generate(**inputs, **model_kwargs)
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)

if __name__ == "__main__":
    gpu = rh.cluster(name='rh-a10x') if rh.exists('rh-a10x') else rh.cluster(name='rh-a10x', instance_type='A100:1')
    flan_t5_generate = rh.function(fn=causal_lm_generate).to(gpu, env=['./'])

    # The first time this runs it will take ~7 minutes to download the model. After that it takes ~4 seconds.
    # Generation options: https://huggingface.co/docs/transformers/main/en/main_classes/text_generation#transformers.GenerationConfig
    my_prompt = "A detailed oil painting of"
    sequences = flan_t5_generate(my_prompt, max_new_tokens=100, min_length=20, temperature=2.0, repetition_penalty=3.0,
                                 use_cache=False, do_sample=True, num_beams=3, num_return_sequences=4,
                                 stream_logs=True)

    sequences = [f"{my_prompt} {seq}" for seq in sequences]
    for seq in sequences:
        print(seq)

    generate_gpu = rh.function(name='sd_generate')
    images = generate_gpu(sequences, num_images=1, steps=50)
    [image.show() for image in images]

    flan_t5_generate.save(name='flan_t5_generate')
