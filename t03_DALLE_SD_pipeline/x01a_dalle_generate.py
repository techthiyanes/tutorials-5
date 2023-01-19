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
    gpu = rh.cluster(name='rh-a10x', instance_type='A100:1')  # On GCP and Azure
    # gpu = rh.cluster(name='rh-a10x', instance_type='g5.2xlarge', provider='aws')  # On AWS

    generate_karlo_gpu = rh.send(fn=unclip_generate, hardware=gpu, name='karlo_generate').save()

    # The model takes a long time to download and send to GPU the first time you run, but after that it only takes
    # 4 seconds per image.
    my_prompt = 'beautiful fantasy painting of Tom Hanks as Samurai in sakura field'
    images = generate_karlo_gpu(my_prompt, num_images=4)
    [image.show() for image in images]