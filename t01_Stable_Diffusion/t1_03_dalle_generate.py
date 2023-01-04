import runhouse as rh
from diffusers import UnCLIPPipeline
import torch


def unclip_generate(prompt,
                    model_id='kakaobrain/karlo-v1-alpha',
                    num_images=1,
                    **model_kwargs):
    pipe = rh.get_pinned_object(model_id)
    if pipe is None:
        pipe = UnCLIPPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to('cuda')
        rh.pin_to_memory(model_id, pipe)
    return pipe([prompt], num_images_per_prompt=num_images, **model_kwargs).images


if __name__ == "__main__":
    # Single A10G is only available on AWS, but you can use an A100 on GCP or Azure instead.
    # See this helpful guide to cloud GPUs for more details: https://www.paperspace.com/gpu-cloud-comparison
    gpu = rh.cluster(name='rh-a10g', instance_type='A10G:1', provider='cheapest')
    generate_karlo_gpu = rh.send(fn=unclip_generate,
                                 hardware=gpu,
                                 reqs=['local:./',
                                       'torch --upgrade --extra-index-url https://download.pytorch.org/whl/cu116',
                                       'diffusers', 'transformers', 'accelerate', 'safetensors'],
                                 name='karlo_generate')

    # The model takes a long time to download and send to GPU the first time you run, but after that it only takes
    # 4 seconds per image.
    my_prompt = 'beautiful fantasy painting of Tom Hanks as Samurai in sakura field'
    images = generate_karlo_gpu(my_prompt, num_images=4)
    [image.show() for image in images]