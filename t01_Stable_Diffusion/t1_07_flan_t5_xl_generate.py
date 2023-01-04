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
    # GCP and Azure
    gpu = rh.cluster(name='rh-a100', instance_type='A100:1', provider='cheapest')
    # AWS, need a g5.2xlarge instance because it has more CPU RAM
    # gpu = rh.cluster(name='rh-a10g', instance_type='g5.2xlarge', provider='aws')
    flan_t5_generate = rh.send(fn=causal_lm_generate,
                               hardware=gpu,
                               reqs=['local:./',
                                     'torch --upgrade --extra-index-url https://download.pytorch.org/whl/cu116',
                                     'transformers'],
                               name='flan_t5_generate')

    # The model takes a long time to download and send to GPU the first time you run, but after that it only takes
    # 4 seconds per image.
    my_prompt = 'My grandmothers recipe for pasta al limone is as follows. Ingredients:'
    # Generation options: https://huggingface.co/docs/transformers/main/en/main_classes/text_generation#transformers.GenerationConfig
    result = flan_t5_generate(my_prompt, max_new_tokens=500, min_length=300, temperature=2.0, repetition_penalty=3.0)
    print(result)
