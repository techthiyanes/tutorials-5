import runhouse as rh
from diffusers import StableDiffusionPipeline
import torch


def sd_generate(prompt, num_images=1, steps=100, guidance_scale=7.5, model_id='stabilityai/stable-diffusion-2-base'):
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16, revision='fp16').to('cuda')
    return pipe([prompt] * num_images, num_inference_steps=steps, guidance_scale=guidance_scale).images


if __name__ == "__main__":
    gpu = rh.cluster(name='rh-a10x') if rh.exists('rh-a10x') else rh.cluster(name='rh-a10x', instance_type='A100:1')
    generate_gpu = rh.function(fn=sd_generate).to(gpu, reqs=['./'])

    images = generate_gpu(prompt='A hot dog made of matcha powder.', num_images=4, steps=50)
    [image.show() for image in images]

    generate_gpu.save(name='sd_generate')

    gpu.keep_warm()
    # gpu.teardown()  # to terminate the cluster immediately
